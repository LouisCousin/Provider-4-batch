"""
Module Batch - Support des traitements par lot OpenAI
=====================================================
Fournit les briques pour soumettre et gérer des batches OpenAI.
"""

import io
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from .core import APIError

# Import des bibliothèques
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


# ============================================================================
# Utilitaires de persistance de l'historique des lots
# ============================================================================

HISTORY_FILE = "batch_history.json"


def _load_local_batch_history() -> List[Dict]:
    """Charge l'historique des lots depuis un fichier JSON local.

    Retourne une liste vide si le fichier est absent ou invalide.
    """
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except (FileNotFoundError, json.JSONDecodeError, TypeError):
        pass
    return []


def _save_batch_to_local_history(batch_id: str, provider: str) -> None:
    """Ajoute un lot soumis à l'historique local sans créer de doublon."""
    history = _load_local_batch_history()

    if any(item.get("id") == batch_id for item in history):
        return

    new_entry = {
        "id": batch_id,
        "provider": provider,
        "submitted_at": datetime.now().isoformat(),
    }
    history.insert(0, new_entry)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
    except Exception:
        pass


# =============================================================================
# Brique n°1 : Requête Batch standardisée
# =============================================================================

@dataclass
class BatchRequest:
    """Représente une requête unique pour l'API Batch d'OpenAI."""
    custom_id: str
    body: Dict
    method: str = "POST"
    url: str = "/v1/chat/completions"

    def __post_init__(self):
        """Validation après initialisation."""
        if not self.custom_id:
            raise ValueError("custom_id ne peut pas être vide")
        if self.method not in ["POST", "GET"]:
            raise ValueError("method doit être POST ou GET")
        if not self.body:
            raise ValueError("body ne peut pas être vide")


# =============================================================================
# Structure standardisée de résultat de batch
# =============================================================================

@dataclass
class BatchResult:
    """Représente un résultat standardisé d'une requête de lot."""

    custom_id: str
    status: str  # 'succeeded' ou 'failed'
    response: Optional[Dict] = None
    error: Optional[Dict] = None
    clean_response: Optional[str] = None
    provider: str = "unknown"
    raw_data: Dict = field(default_factory=dict)


# =============================================================================
# Brique n°2 : Mixin pour la soumission de batches
# =============================================================================

class OpenAIBatchMixin:
    """
    Mixin pour ajouter la fonctionnalité de soumission de batch
    aux providers compatibles OpenAI.
    """
    
    def submit_batch(self, requests: List[BatchRequest], metadata: Dict = None) -> str:
        """
        Prépare, uploade et soumet un travail en lot à l'API OpenAI.
        Retourne l'ID du batch.
        
        Args:
            requests: Liste de BatchRequest à soumettre
            metadata: Métadonnées optionnelles pour le batch
            
        Returns:
            str: ID du batch créé
            
        Raises:
            APIError: En cas d'erreur lors de la soumission
        """
        if not hasattr(self, 'client') or not self.client:
            raise APIError("Le client OpenAI n'est pas initialisé.")
        
        if not requests:
            raise ValueError("La liste de requêtes ne peut pas être vide")
        
        # 1. Conversion en JSONL
        jsonl_lines = []
        for req in requests:
            batch_line = {
                "custom_id": req.custom_id,
                "method": req.method,
                "url": req.url,
                "body": req.body
            }
            jsonl_lines.append(json.dumps(batch_line))
        
        jsonl_content = "\n".join(jsonl_lines)
        file_obj = io.BytesIO(jsonl_content.encode('utf-8'))
        
        # 2. Upload du fichier
        try:
            uploaded_file = self.client.files.create(
                file=file_obj,
                purpose="batch"
            )
            print(f"✅ Fichier batch uploadé: {uploaded_file.id}")
        except Exception as e:
            raise APIError(f"Échec de l'upload du fichier batch: {e}")
        
        # 3. Création du batch
        try:
            batch_job = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata=metadata or {}
            )
            print(f"✅ Batch créé avec succès: {batch_job.id}")
            _save_batch_to_local_history(batch_job.id, "openai")
            return batch_job.id
        except Exception as e:
            raise APIError(f"Échec de la création du batch: {e}")


# =============================================================================
# Brique n°2b : Mixin pour la soumission de batches Anthropic
# =============================================================================

class AnthropicBatchMixin:
    """
    Mixin pour ajouter la fonctionnalité de soumission de batch
    aux providers Anthropic (Claude).
    """
    
    def submit_batch(self, requests: List[BatchRequest], metadata: Dict = None) -> str:
        """
        Prépare, uploade et soumet un travail en lot à l'API Anthropic.
        Retourne l'ID du batch.
        
        Args:
            requests: Liste de BatchRequest à soumettre
            metadata: Métadonnées optionnelles pour le batch (non utilisé par Anthropic)
            
        Returns:
            str: ID du batch créé
            
        Raises:
            APIError: En cas d'erreur lors de la soumission
        """
        if not hasattr(self, 'client') or not self.client:
            raise APIError("Le client Anthropic n'est pas initialisé.")
        
        if not requests:
            raise ValueError("La liste de requêtes ne peut pas être vide")
        
        # Convertir les BatchRequest au format Anthropic
        batch_requests = []
        for req in requests:
            # Anthropic utilise un format différent
            anthropic_request = {
                "custom_id": req.custom_id,
                "params": {
                    "model": req.body.get("model", self.model_name),
                    "messages": req.body.get("messages", []),
                    "max_tokens": req.body.get("max_tokens", 1000)
                }
            }
            
            # Ajouter les paramètres optionnels s'ils existent
            for param in ["temperature", "top_p", "top_k", "stop_sequences"]:
                if param in req.body:
                    anthropic_request["params"][param] = req.body[param]
            
            batch_requests.append(anthropic_request)
        
        try:
            # Créer le batch via l'API Anthropic
            batch = self.client.beta.messages.batches.create(
                requests=batch_requests
            )

            print(f"✅ Batch Anthropic créé avec succès: {batch.id}")
            _save_batch_to_local_history(batch.id, "anthropic")
            return batch.id

        except Exception as e:
            raise APIError(f"Échec de la création du batch Anthropic: {e}")


# =============================================================================
# Brique n°3 : Gestionnaire de tâches Batch (OpenAI et Anthropic)
# =============================================================================

class BatchJobManager:
    """
    Brique dédiée à la gestion des batches (OpenAI et Anthropic).
    Supporte les deux APIs avec détection automatique.
    """
    
    def __init__(self, api_key: str, provider_type: str = "openai"):
        """
        Initialise le gestionnaire avec une clé API.
        
        Args:
            api_key: Clé API (OpenAI ou Anthropic)
            provider_type: Type de provider ("openai" ou "anthropic")
        """
        self.api_key = api_key
        self.provider_type = provider_type.lower()
        self.client = None
        
        if api_key and api_key.strip():
            try:
                if self.provider_type == "anthropic":
                    if anthropic is None:
                        raise ImportError("Installez anthropic: pip install anthropic")
                    self.client = anthropic.Anthropic(api_key=api_key)
                    print(f"✅ BatchJobManager initialisé (Anthropic)")
                else:  # Default to OpenAI
                    if openai is None:
                        raise ImportError("Installez openai: pip install openai")
                    self.client = openai.OpenAI(api_key=api_key)
                    self.provider_type = "openai"
                    print(f"✅ BatchJobManager initialisé (OpenAI)")
            except Exception as e:
                print(f"❌ Erreur initialisation BatchJobManager: {e}")
                self.client = None

    def _unify_status(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Ajoute un statut unifié à un dictionnaire d'informations de lot.

        Normalise les valeurs spécifiques au provider pour exposer un ensemble
        commun de statuts : ``running``, ``completed``, ``failed`` ou
        ``unknown``. Le champ de statut original est conservé pour les besoins
        de débogage.
        """

        unified_status = "unknown"
        provider = batch_info.get('provider', self.provider_type)
        raw_status = batch_info.get('status')

        if provider == "anthropic":
            status_map = {
                "ended": "completed",
                "processing": "running",
                "created": "running",
                "expired": "failed",
                "canceling": "failed",
            }
            unified_status = status_map.get(raw_status, "unknown")
        else:  # OpenAI par défaut
            status_map = {
                "completed": "completed",
                "validating": "running",
                "in_progress": "running",
                "failed": "failed",
                "expired": "failed",
                "cancelled": "failed",
            }
            unified_status = status_map.get(raw_status, "unknown")

        batch_info['unified_status'] = unified_status
        return batch_info

    def _extract_request_counts(self, rc: Any, provider: str) -> Optional[Dict[str, Any]]:
        """Normalise l'objet ``request_counts`` selon le provider."""
        if not rc:
            return None

        def _get(obj: Any, name: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        counts = {
            'total': _get(rc, 'total'),
            'processing': _get(rc, 'processing'),
            'succeeded': _get(rc, 'succeeded'),
            'errored': _get(rc, 'errored'),
            'canceled': _get(rc, 'canceled'),
        }

        if provider != 'anthropic':
            # OpenAI utilise 'completed' et 'failed'
            completed = _get(rc, 'completed')
            failed = _get(rc, 'failed')
            if completed is not None:
                counts['succeeded'] = completed
            if failed is not None:
                counts['errored'] = failed

        # Retirer les entrées None pour alléger la structure
        return {k: v for k, v in counts.items() if v is not None}

    def get_history(self, limit: int = 50) -> List[Dict]:
        """
        Récupère l'historique des batches.
        
        Args:
            limit: Nombre maximum de batches à récupérer
            
        Returns:
            List[Dict]: Liste des batches avec leurs métadonnées
        """
        if not self.client:
            return []
        
        try:
            if self.provider_type == "anthropic":
                # API Anthropic pour lister les batches
                batches = self.client.beta.messages.batches.list(limit=limit)
                batch_list = []

                for batch in batches.data:
                    batch_info = {
                        'id': batch.id,
                        'status': batch.processing_status,
                        'processing_status': batch.processing_status,
                        'created_at': batch.created_at,
                        'request_counts': self._extract_request_counts(
                            getattr(batch, 'request_counts', None),
                            'anthropic'
                        ),
                        'provider': 'anthropic'
                    }
                    batch_list.append(batch_info)
            else:
                # API OpenAI pour lister les batches
                batches = self.client.batches.list(limit=limit)
                batch_list = []

                for batch in batches.data:
                    batch_info = {
                        'id': batch.id,
                        'status': batch.status,
                        'created_at': datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                        'endpoint': batch.endpoint,
                        'completion_window': batch.completion_window,
                        'request_counts': self._extract_request_counts(
                            getattr(batch, 'request_counts', None),
                            'openai'
                        ),
                        'output_file_id': getattr(batch, 'output_file_id', None),
                        'error_file_id': getattr(batch, 'error_file_id', None),
                        'metadata': getattr(batch, 'metadata', {}),
                        'provider': 'openai'
                    }
                    batch_list.append(batch_info)

            return [self._unify_status(batch) for batch in batch_list]

        except Exception as e:
            print(f"❌ Erreur récupération historique: {str(e)}")
            return []
    
    def get_status(self, batch_id: str) -> Optional[Dict]:
        """
        Récupère le statut détaillé d'un batch spécifique.
        
        Args:
            batch_id: ID du batch
            
        Returns:
            Optional[Dict]: Informations du batch ou None si non trouvé
        """
        if not self.client:
            return None
        
        try:
            if self.provider_type == "anthropic":
                # API Anthropic
                batch = self.client.beta.messages.batches.retrieve(batch_id)

                batch_info = {
                    'id': batch.id,
                    'status': batch.processing_status,
                    'processing_status': batch.processing_status,
                    'created_at': batch.created_at,
                    'expires_at': batch.expires_at,
                    'request_counts': self._extract_request_counts(
                        getattr(batch, 'request_counts', None),
                        'anthropic'
                    ),
                    'results_url': getattr(batch, 'results_url', None),
                    'provider': 'anthropic'
                }
            else:
                # API OpenAI
                if not batch_id.startswith('batch_'):
                    return None

                batch = self.client.batches.retrieve(batch_id)

                batch_info = {
                    'id': batch.id,
                    'status': batch.status,
                    'created_at': datetime.fromtimestamp(batch.created_at).strftime("%Y-%m-%d %H:%M:%S"),
                    'endpoint': batch.endpoint,
                    'completion_window': batch.completion_window,
                    'request_counts': self._extract_request_counts(
                        getattr(batch, 'request_counts', None),
                        'openai'
                    ),
                    'output_file_id': getattr(batch, 'output_file_id', None),
                    'error_file_id': getattr(batch, 'error_file_id', None),
                    'input_file_id': getattr(batch, 'input_file_id', None),
                    'metadata': getattr(batch, 'metadata', {}),
                    'provider': 'openai'
                }

            return self._unify_status(batch_info)

        except Exception as e:
            print(f"❌ Erreur recherche batch {batch_id}: {str(e)}")
            return None
    
    def get_results(self, batch_id: str) -> List[BatchResult]:
        """Télécharge et parse les résultats d'un batch.

        Chaque ligne des fichiers de sortie ou d'erreur est convertie en
        ``BatchResult`` décrivant soit une réussite, soit une erreur pour la
        requête identifiée par ``custom_id``.
        """

        if not self.client:
            return []

        try:
            results: List[BatchResult] = []

            if self.provider_type == "anthropic":
                batch = self.client.beta.messages.batches.retrieve(batch_id)
                if batch.processing_status != "ended":
                    print(
                        f"⚠️ Batch {batch_id} non terminé (statut: {batch.processing_status})"
                    )
                    return []

                for result in self.client.beta.messages.batches.results(batch_id):
                    if result.result.type == "succeeded":
                        message = result.result.message
                        content = ""
                        if getattr(message, "content", None):
                            first = message.content[0]
                            content = (
                                first.text if hasattr(first, "text") else first.get("text", "")
                            )

                        response = {"content": content, "role": message.role}
                        raw = result.model_dump() if hasattr(result, "model_dump") else {}
                        results.append(
                            BatchResult(
                                custom_id=result.custom_id,
                                status="succeeded",
                                response=response,
                                clean_response=content,
                                provider="anthropic",
                                raw_data=raw,
                            )
                        )
                    elif result.result.type == "errored":
                        error_obj = getattr(result.result, "error", None)
                        error = (
                            error_obj.model_dump()
                            if hasattr(error_obj, "model_dump")
                            else getattr(error_obj, "__dict__", {"message": str(error_obj)})
                        )
                        raw = result.model_dump() if hasattr(result, "model_dump") else {}
                        results.append(
                            BatchResult(
                                custom_id=result.custom_id,
                                status="failed",
                                error=error,
                                provider="anthropic",
                                raw_data=raw,
                            )
                        )

                return results

            # Provider OpenAI par défaut
            batch = self.client.batches.retrieve(batch_id)
            if batch.status != "completed":
                print(f"⚠️ Batch {batch_id} non terminé (statut: {batch.status})")
                return []

            if getattr(batch, "output_file_id", None):
                success_content = self.client.files.content(batch.output_file_id).text
                for line in success_content.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    response_body = data.get("response", {}).get("body", {})
                    clean_content = None
                    if isinstance(response_body, dict):
                        choices = response_body.get("choices")
                        if choices:
                            first_choice = choices[0]
                            if isinstance(first_choice, dict):
                                message = first_choice.get("message", {})
                                if isinstance(message, dict):
                                    clean_content = message.get("content")
                    results.append(
                        BatchResult(
                            custom_id=data.get("custom_id"),
                            status="succeeded",
                            response=response_body,
                            clean_response=clean_content,
                            provider="openai",
                            raw_data=data,
                        )
                    )

            if getattr(batch, "error_file_id", None):
                error_content = self.client.files.content(batch.error_file_id).text
                for line in error_content.strip().split("\n"):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    results.append(
                        BatchResult(
                            custom_id=data.get("custom_id"),
                            status="failed",
                            error=data.get("response", {}).get("body"),
                            provider="openai",
                            raw_data=data,
                        )
                    )

            return results

        except Exception as e:
            print(f"❌ Erreur téléchargement résultats: {str(e)}")
            return []
    
    def wait_for_completion(self, batch_id: str, poll_interval: int = 60, timeout: int = 86400) -> Dict:
        """
        Attend qu'un batch soit terminé en interrogeant son statut.
        
        Args:
            batch_id: ID du batch à surveiller
            poll_interval: Intervalle entre les vérifications (secondes)
            timeout: Temps maximum d'attente (secondes, défaut 24h)
            
        Returns:
            Dict: Statut final du batch
        """
        if not self.client:
            raise APIError("Client non initialisé")
        
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise APIError(f"Timeout: le batch {batch_id} n'est pas terminé après {timeout}s")
            
            status = self.get_status(batch_id)
            if not status:
                raise APIError(f"Impossible de récupérer le statut du batch {batch_id}")
            
            if self.provider_type == "anthropic":
                current_status = status['status']
                print(f"📊 Batch {batch_id}: {current_status} (temps écoulé: {int(elapsed)}s)")
                
                if current_status == 'ended':
                    print(f"✅ Batch {batch_id} terminé!")
                    return status
                elif current_status in ['canceling', 'expired']:
                    raise APIError(f"Batch {batch_id} terminé avec erreur: {current_status}")
            else:
                current_status = status['status']
                print(f"📊 Batch {batch_id}: {current_status} (temps écoulé: {int(elapsed)}s)")
                
                if current_status == 'completed':
                    print(f"✅ Batch {batch_id} terminé avec succès!")
                    return status
                elif current_status in ['failed', 'expired', 'cancelled']:
                    raise APIError(f"Batch {batch_id} terminé avec erreur: {current_status}")
            
            # Attendre avant la prochaine vérification
            time.sleep(poll_interval)
    
    def cancel_batch(self, batch_id: str) -> bool:
        """
        Annule un batch en cours.
        
        Args:
            batch_id: ID du batch à annuler
            
        Returns:
            bool: True si annulé avec succès
        """
        if not self.client:
            return False
        
        try:
            if self.provider_type == "anthropic":
                self.client.beta.messages.batches.cancel(batch_id)
                print(f"✅ Batch Anthropic {batch_id} annulé")
            else:
                self.client.batches.cancel(batch_id)
                print(f"✅ Batch OpenAI {batch_id} annulé")
            return True
        except Exception as e:
            print(f"❌ Erreur annulation batch: {e}")
            return False
