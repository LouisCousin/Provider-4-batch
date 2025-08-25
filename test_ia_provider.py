"""
Tests unitaires pour le module ia_provider
==========================================
Tests pour gpt-4.1, claude-sonnet-4-20250514 et GPT-5
"""

import pytest
import os
import json
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
from types import SimpleNamespace

# Import du module à tester
from ia_provider import (
    BaseProvider, 
    ProviderManager, 
    APIError, 
    UnknownModelError,
    load_config, 
    load_api_key, 
    manager
)
from ia_provider.openai import OpenAIProvider
from ia_provider.anthropic import AnthropicProvider
from ia_provider.gpt5 import GPT5Provider
from ia_provider.batch import BatchJobManager, BatchResult


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Configuration de test par défaut."""
    return {
        'temperature': 0.7,
        'max_tokens': 1000,
        'top_p': 0.95,
        'top_k': 40,
        'frequency_penalty': 0,
        'presence_penalty': 0
    }


@pytest.fixture
def sample_messages():
    """Messages de test pour les conversations."""
    return [
        {"role": "user", "content": "Bonjour"},
        {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
        {"role": "user", "content": "Quelle est la capitale de la France?"}
    ]


# =============================================================================
# Tests de normalisation des statuts de batch
# =============================================================================

class TestBatchStatusUnification:
    @pytest.mark.parametrize("raw,expected", [
        ("in_progress", "running"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "failed"),
    ])
    def test_unify_status_openai_mapping(self, raw, expected):
        manager = BatchJobManager(api_key="", provider_type="openai")
        result = manager._unify_status({"status": raw, "provider": "openai"})
        assert result["unified_status"] == expected
        assert result["status"] == raw

    @pytest.mark.parametrize("raw,expected", [
        ("processing", "running"),
        ("ended", "completed"),
        ("expired", "failed"),
    ])
    def test_unify_status_anthropic_mapping(self, raw, expected):
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        result = manager._unify_status({"status": raw, "provider": "anthropic"})
        assert result["unified_status"] == expected
        assert result["status"] == raw

    def test_get_status_and_history_include_unified_status_openai(self):
        manager = BatchJobManager(api_key="", provider_type="openai")

        batch_status = SimpleNamespace(
            id="batch_b1",
            status="in_progress",
            created_at=0,
            endpoint="/v1/chat",
            completion_window="24h",
            request_counts=None,
            output_file_id=None,
            error_file_id=None,
            input_file_id=None,
            metadata={},
        )

        batch_history = SimpleNamespace(
            id="b2",
            status="completed",
            created_at=0,
            endpoint="/v1/chat",
            completion_window="24h",
            request_counts=None,
            output_file_id=None,
            error_file_id=None,
            metadata={},
        )

        manager.client = SimpleNamespace(
            batches=SimpleNamespace(
                retrieve=lambda batch_id: batch_status,
                list=lambda limit: SimpleNamespace(data=[batch_history]),
            )
        )

        status = manager.get_status("batch_b1")
        assert status["status"] == "in_progress"
        assert status["unified_status"] == "running"

        history = manager.get_history(limit=1)
        assert len(history) == 1
        assert history[0]["status"] == "completed"
        assert history[0]["unified_status"] == "completed"
        assert "unified_status" in status and "unified_status" in history[0]


class TestAnthropicBatchManagement:
    """Tests de gestion des lots pour Anthropic."""

    def test_get_history(self):
        batch = SimpleNamespace(
            id="b1",
            processing_status="processing",
            created_at="2024-01-01",
            request_counts=SimpleNamespace(
                total=1, processing=1, succeeded=0, errored=0, canceled=0
            ),
        )
        list_mock = MagicMock(return_value=SimpleNamespace(data=[batch]))
        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(batches=SimpleNamespace(list=list_mock))
            )
        )
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        manager.client = client

        history = manager.get_history(limit=5)
        list_mock.assert_called_once_with(limit=5)
        assert history[0]["id"] == "b1"
        assert history[0]["unified_status"] == "running"

    def test_get_status(self):
        batch_obj = SimpleNamespace(
            id="b1",
            processing_status="ended",
            created_at="2024-01-01",
            expires_at="2024-01-02",
            request_counts=SimpleNamespace(
                total=1, processing=0, succeeded=1, errored=0, canceled=0
            ),
            results_url="http://example",
        )
        retrieve_mock = MagicMock(return_value=batch_obj)
        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(batches=SimpleNamespace(retrieve=retrieve_mock))
            )
        )
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        manager.client = client

        status = manager.get_status("b1")
        retrieve_mock.assert_called_once_with("b1")
        assert status["unified_status"] == "completed"
        assert status["id"] == "b1"

    def test_get_results(self):
        retrieve_mock = MagicMock(return_value=SimpleNamespace(processing_status="ended"))

        success_result = SimpleNamespace(
            custom_id="1",
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(
                    role="assistant", content=[SimpleNamespace(text="ok")]
                ),
            ),
            model_dump=lambda: {"id": 1},
        )
        error_obj = SimpleNamespace(message="boom", model_dump=lambda: {"message": "boom"})
        error_result = SimpleNamespace(
            custom_id="2",
            result=SimpleNamespace(type="errored", error=error_obj),
            model_dump=lambda: {"id": 2},
        )
        results_mock = MagicMock(return_value=[success_result, error_result])

        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(
                    batches=SimpleNamespace(
                        retrieve=retrieve_mock, results=results_mock
                    )
                )
            )
        )
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        manager.client = client

        results = manager.get_results("b1")
        retrieve_mock.assert_called_once_with("b1")
        results_mock.assert_called_once_with("b1")
        assert len(results) == 2
        assert results[0].status == "succeeded"
        assert results[0].response["content"] == "ok"
        assert results[1].status == "failed"
        assert results[1].error["message"] == "boom"

    def test_cancel_batch(self):
        cancel_mock = MagicMock()
        client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(batches=SimpleNamespace(cancel=cancel_mock))
            )
        )
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        manager.client = client

        assert manager.cancel_batch("b1") is True
        cancel_mock.assert_called_once_with("b1")


# =============================================================================
# Tests du ProviderManager
# =============================================================================

class TestProviderManager:
    """Tests pour le gestionnaire de providers."""
    
    def test_available_models(self):
        """Test que les modèles attendus sont disponibles."""
        models = manager.get_available_models()

        assert "gpt-4.1" in models
        assert "claude-sonnet-4-20250514" in models

        # Vérifier que certains modèles non supportés ne sont pas présents
        assert "gpt-4" not in models
        assert "claude-3-opus" not in models
        assert "gemini-pro" not in models
    
    @patch('ia_provider.openai.openai')
    def test_get_provider_gpt41(self, mock_openai):
        """Test de récupération du provider GPT-4.1."""
        provider = manager.get_provider("gpt-4.1", "test-key")
        
        assert isinstance(provider, OpenAIProvider)
        assert provider.model_name == "gpt-4.1"
        assert provider.api_key == "test-key"
    
    @patch('ia_provider.anthropic.anthropic')
    def test_get_provider_claude(self, mock_anthropic):
        """Test de récupération du provider Claude."""
        provider = manager.get_provider("claude-sonnet-4-20250514", "test-key")

        assert isinstance(provider, AnthropicProvider)
        assert provider.model_name == "claude-sonnet-4-20250514"
        assert provider.api_key == "test-key"
    
    def test_get_provider_unknown_model(self):
        """Test avec un modèle non supporté."""
        with pytest.raises(UnknownModelError, match="gpt-4"):
            manager.get_provider("gpt-4", "key")
        
        with pytest.raises(UnknownModelError, match="claude-3-opus"):
            manager.get_provider("claude-3-opus", "key")


# =============================================================================
# Tests du provider OpenAI (GPT-4.1)
# =============================================================================

class TestOpenAIProvider:
    """Tests pour le provider GPT-4.1."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock du module openai."""
        with patch('ia_provider.openai.openai') as mock:
            mock.OpenAI = MagicMock()
            yield mock
    
    def test_max_completion_tokens_conversion(self, mock_openai):
        """Test que max_tokens est converti en max_completion_tokens."""
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
        provider.client.chat.completions.create.return_value = mock_response
        
        # Appel avec max_tokens
        provider.generer_reponse("Test prompt", max_tokens=500)
        
        # Vérifier la conversion
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert 'max_completion_tokens' in call_kwargs
        assert call_kwargs['max_completion_tokens'] == 500
        assert 'max_tokens' not in call_kwargs
    
    def test_generer_reponse_success(self, mock_openai):
        """Test de génération réussie."""
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Réponse GPT-4.1"))]
        provider.client.chat.completions.create.return_value = mock_response
        
        response = provider.generer_reponse(
            "Test prompt",
            temperature=0.5,
            max_tokens=200
        )
        
        assert response == "Réponse GPT-4.1"
        
        # Vérifier les paramètres
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs['model'] == "gpt-4.1"
        assert call_kwargs['max_completion_tokens'] == 200
        assert 'temperature' in call_kwargs
    
    def test_parameter_filtering(self, mock_openai):
        """Test que les paramètres non supportés sont filtrés."""
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
        provider.client.chat.completions.create.return_value = mock_response
        
        # Appel avec paramètres mixtes (supportés et non supportés)
        provider.generer_reponse(
            "Test",
            temperature=0.5,
            max_tokens=100,
            top_k=40,  # Non supporté par OpenAI
            frequency_penalty=0.5,  # Supporté
            custom_param="test"  # Non supporté
        )
        
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        
        # Vérifier le filtrage
        assert 'top_k' not in call_kwargs
        assert 'custom_param' not in call_kwargs
        assert 'frequency_penalty' in call_kwargs
        assert call_kwargs['frequency_penalty'] == 0.5
        """Test de conversation avec historique."""
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Paris"))]
        provider.client.chat.completions.create.return_value = mock_response
        
        response = provider.chatter(sample_messages, max_tokens=100)
        
        assert response == "Paris"
        
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        assert call_kwargs['messages'] == sample_messages
        assert call_kwargs['max_completion_tokens'] == 100
    
    def test_parameter_filtering(self, mock_openai):
        """Test que les paramètres non supportés sont filtrés."""
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test"))]
        provider.client.chat.completions.create.return_value = mock_response
        
        # Appel avec paramètres mixtes (supportés et non supportés)
        provider.generer_reponse(
            "Test",
            temperature=0.5,
            max_tokens=100,
            top_k=40,  # Non supporté par OpenAI
            frequency_penalty=0.5,  # Supporté
            custom_param="test"  # Non supporté
        )
        
        call_kwargs = provider.client.chat.completions.create.call_args[1]
        
        # Vérifier le filtrage
        assert 'top_k' not in call_kwargs
        assert 'custom_param' not in call_kwargs
        assert 'frequency_penalty' in call_kwargs
        assert call_kwargs['frequency_penalty'] == 0.5


# =============================================================================
# Tests du provider GPT-5
# =============================================================================

class TestGPT5Provider:
    """Tests pour le provider GPT-5."""

    @patch('ia_provider.gpt5.openai')
    def test_gpt5_nano_final_fix(self, mock_openai):
        """Vérifie que gpt-5-nano force les bons paramètres."""

        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test OK"))]
        mock_client.chat.completions.create.return_value = mock_response

        provider = GPT5Provider("gpt-5-nano", "test-key")

        provider.generer_reponse(
            "Test",
            reasoning_effort="high",  # Doit être ignoré
            verbosity="high",
            max_tokens=123,
            temperature=0.7  # Ce paramètre devrait être ignoré
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs.get('reasoning_effort') == 'minimal'
        assert 'max_tokens' not in call_kwargs
        assert call_kwargs.get('max_completion_tokens') == 123
        assert 'temperature' not in call_kwargs

# =============================================================================
# Tests du provider Anthropic (Claude Sonnet 4)
# =============================================================================

class TestAnthropicProvider:
    """Tests pour le provider Claude Sonnet 4."""

    @pytest.fixture
    def mock_anthropic(self):
        """Mock du module anthropic."""
        with patch('ia_provider.anthropic.anthropic') as mock:
            mock.Anthropic = MagicMock()
            yield mock

    def test_thinking_parameter(self, mock_anthropic):
        """Test du support du mode thinking."""
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Réponse avec thinking")]
        provider.client.messages.create.return_value = mock_response

        # Test avec thinking_budget
        provider.generer_reponse(
            "Test complexe",
            thinking_budget=300,
            max_tokens=500
        )

        call_kwargs = provider.client.messages.create.call_args[1]
        assert 'thinking' in call_kwargs
        assert call_kwargs['thinking']['type'] == 'enabled'
        assert call_kwargs['thinking']['budget_tokens'] == 300

    def test_max_tokens_always_present(self, mock_anthropic):
        """Test que max_tokens est toujours ajouté."""
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test")]
        provider.client.messages.create.return_value = mock_response

        # Appel sans spécifier max_tokens
        provider.generer_reponse("Test")

        call_kwargs = provider.client.messages.create.call_args[1]
        assert 'max_tokens' in call_kwargs
        assert call_kwargs['max_tokens'] == 1000  # Valeur par défaut

    def test_generer_reponse_success(self, mock_anthropic):
        """Test de génération réussie."""
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Réponse Claude")]
        provider.client.messages.create.return_value = mock_response

        response = provider.generer_reponse(
            "Test prompt",
            temperature=0.6,
            max_tokens=250
        )

        assert response == "Réponse Claude"

        call_kwargs = provider.client.messages.create.call_args[1]
        assert call_kwargs['model'] == "claude-sonnet-4-20250514"
        assert call_kwargs['max_tokens'] == 250

    def test_chatter_validates_roles(self, mock_anthropic):
        """Test de validation des rôles."""
        provider = AnthropicProvider("claude-sonnet-4-20250514", "test-key")

        invalid_messages = [{"role": "system", "content": "Test"}]

        with pytest.raises(ValueError, match="'user' ou 'assistant'"):
            provider.chatter(invalid_messages)


# =============================================================================
# Tests d'intégration
# =============================================================================

class TestIntegration:
    """Tests d'intégration du système."""
    
    def test_both_models_registered(self):
        """Test que les deux modèles sont bien enregistrés."""
        providers_info = manager.get_providers_info()
        
        assert 'OpenAIProvider' in providers_info
        assert 'AnthropicProvider' in providers_info
        
        assert 'gpt-4.1' in providers_info['OpenAIProvider']
        assert 'claude-sonnet-4-20250514' in providers_info['AnthropicProvider']
    
    @patch('ia_provider.openai.openai')
    @patch('ia_provider.anthropic.anthropic')
    def test_switching_providers(self, mock_anthropic, mock_openai):
        """Test du changement de provider."""
        # Test GPT-4.1
        provider1 = manager.get_provider("gpt-4.1", "key1")
        assert isinstance(provider1, OpenAIProvider)
        
        # Test Claude
        provider2 = manager.get_provider("claude-sonnet-4-20250514", "key2")
        assert isinstance(provider2, AnthropicProvider)
        
        # Vérifier que ce sont bien des instances différentes
        assert provider1 != provider2


# =============================================================================
# Tests des fonctions utilitaires
# =============================================================================

class TestUtilities:
    """Tests des fonctions utilitaires."""
    
    def test_load_config_default(self):
        """Test du chargement de configuration par défaut."""
        with patch('os.path.exists', return_value=False):
            config = load_config()
            assert config['temperature'] == 0.7
            assert config['max_tokens'] == 1000
    
    def test_load_api_key_from_env(self):
        """Test du chargement de clé API."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            key = load_api_key('OpenAIProvider')
            assert key == 'test-key'
    
    def test_load_api_key_missing(self):
        """Test avec clé manquante."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Clé API non trouvée"):
                load_api_key('OpenAIProvider')


# =============================================================================
# Tests de persistance de l'historique des lots
# =============================================================================

class TestBatchHistoryPersistence:
    """Tests pour la sauvegarde et le chargement de l'historique local."""

    def test_save_and_load_history(self, tmp_path, monkeypatch):
        import ia_provider.batch as batch_module
        history_file = tmp_path / "batch_history.json"
        monkeypatch.setattr(batch_module, "HISTORY_FILE", str(history_file))

        batch_module._save_batch_to_local_history("b123", "openai")
        history = batch_module._load_local_batch_history()
        assert len(history) == 1
        assert history[0]["id"] == "b123"
        assert history[0]["provider"] == "openai"
        assert "submitted_at" in history[0]

        # Vérifier l'absence de duplication
        batch_module._save_batch_to_local_history("b123", "openai")
        assert len(batch_module._load_local_batch_history()) == 1

    def test_load_history_returns_empty_on_error(self, tmp_path, monkeypatch):
        import ia_provider.batch as batch_module
        history_file = tmp_path / "batch_history.json"
        monkeypatch.setattr(batch_module, "HISTORY_FILE", str(history_file))

        # Fichier manquant
        assert batch_module._load_local_batch_history() == []

        # Fichier JSON invalide
        history_file.write_text("{invalid")
        assert batch_module._load_local_batch_history() == []


class TestSubmitBatchPersists:
    """Tests que les mixins sauvegardent l'historique lors de la soumission."""

    def test_openai_submit_batch_calls_save(self, monkeypatch):
        import ia_provider.batch as batch_module

        class Dummy(batch_module.OpenAIBatchMixin):
            pass

        dummy = Dummy()
        dummy.client = SimpleNamespace(
            files=SimpleNamespace(create=lambda file, purpose: SimpleNamespace(id="file_1")),
            batches=SimpleNamespace(
                create=lambda input_file_id, endpoint, completion_window, metadata: SimpleNamespace(id="batch_1")
            ),
        )

        called = {}

        def fake_save(batch_id, provider):
            called["id"] = batch_id
            called["provider"] = provider

        monkeypatch.setattr(batch_module, "_save_batch_to_local_history", fake_save)

        req = batch_module.BatchRequest(custom_id="1", body={"model": "gpt-4.1", "messages": []})
        returned_id = dummy.submit_batch([req])
        assert returned_id == "batch_1"
        assert called == {"id": "batch_1", "provider": "openai"}

    def test_anthropic_submit_batch_calls_save(self, monkeypatch):
        import ia_provider.batch as batch_module

        class Dummy(batch_module.AnthropicBatchMixin):
            pass

        dummy = Dummy()
        dummy.model_name = "claude"
        dummy.client = SimpleNamespace(
            beta=SimpleNamespace(
                messages=SimpleNamespace(
                    batches=SimpleNamespace(
                        create=lambda requests: SimpleNamespace(id="batch_a1")
                    )
                )
            )
        )

        called = {}

        def fake_save(batch_id, provider):
            called["id"] = batch_id
            called["provider"] = provider

        monkeypatch.setattr(batch_module, "_save_batch_to_local_history", fake_save)

        req = batch_module.BatchRequest(custom_id="1", body={"messages": [], "model": "claude", "max_tokens": 10})
        returned_id = dummy.submit_batch([req])
        assert returned_id == "batch_a1"
        assert called == {"id": "batch_a1", "provider": "anthropic"}


# =============================================================================
# Tests pour la récupération des résultats de batch
# =============================================================================


class TestBatchGetResults:
    """Vérifie que get_results retourne des ``BatchResult`` normalisés."""

    # Les tests suivants s'assurent que chaque ligne de sortie d'un batch est
    # transformée en instance de ``BatchResult`` comportant l'état et les
    # données brutes associées.

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------
    @staticmethod
    def _ns(**kwargs):
        obj = SimpleNamespace(**kwargs)
        obj.model_dump = lambda: obj.__dict__
        return obj

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    def _openai_manager(self, output_text="", error_text="", output_file_id=None, error_file_id=None):
        manager = BatchJobManager(api_key="", provider_type="openai")
        batch = SimpleNamespace(
            status="completed",
            output_file_id=output_file_id,
            error_file_id=error_file_id,
        )

        def content(file_id):
            if file_id == output_file_id:
                return SimpleNamespace(text=output_text)
            if file_id == error_file_id:
                return SimpleNamespace(text=error_text)
            return SimpleNamespace(text="")

        manager.client = SimpleNamespace(
            batches=SimpleNamespace(retrieve=lambda batch_id: batch),
            files=SimpleNamespace(content=content),
        )
        return manager

    def test_openai_success_error_mixed(self):
        success_lines = "\n".join(
            [
                json.dumps({"custom_id": "s1", "response": {"body": {"ok": 1}}}),
                json.dumps({"custom_id": "s2", "response": {"body": {"ok": 2}}}),
            ]
        )
        error_lines = "\n".join(
            [
                json.dumps({"custom_id": "e1", "response": {"body": {"error": "boom"}}}),
                json.dumps({"custom_id": "e2", "response": {"body": {"error": "bad"}}}),
            ]
        )

        # Succès uniquement
        mgr = self._openai_manager(output_text=success_lines, output_file_id="out")
        res = mgr.get_results("batch")
        assert [r.status for r in res] == ["succeeded", "succeeded"]
        assert all(isinstance(r, BatchResult) for r in res)

        # Erreurs uniquement
        mgr = self._openai_manager(error_text=error_lines, error_file_id="err")
        res = mgr.get_results("batch")
        assert [r.status for r in res] == ["failed", "failed"]
        assert res[0].error == {"error": "boom"}

        # Mixte
        mgr = self._openai_manager(
            output_text=success_lines.split("\n")[0],
            error_text=error_lines.split("\n")[0],
            output_file_id="out",
            error_file_id="err",
        )
        res = mgr.get_results("batch")
        assert len(res) == 2
        assert {r.status for r in res} == {"succeeded", "failed"}

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    def _anthropic_manager(self, results_list):
        manager = BatchJobManager(api_key="", provider_type="anthropic")
        batch = SimpleNamespace(processing_status="ended")

        manager.client = self._ns(
            beta=self._ns(
                messages=self._ns(
                    batches=self._ns(
                        retrieve=lambda batch_id: batch,
                        results=lambda batch_id: results_list,
                    )
                )
            )
        )
        return manager

    def _success_result(self, custom_id):
        message = self._ns(
            content=[self._ns(text="ok")],
            role="assistant",
            model="claude",
            usage=self._ns(input_tokens=1, output_tokens=1),
        )
        inner = self._ns(type="succeeded", message=message)
        return self._ns(custom_id=custom_id, result=inner)

    def _error_result(self, custom_id):
        err = self._ns(message="bad request")
        inner = self._ns(type="errored", error=err)
        return self._ns(custom_id=custom_id, result=inner)

    def test_anthropic_success_error_mixed(self):
        # Succès uniquement
        mgr = self._anthropic_manager([self._success_result("a1"), self._success_result("a2")])
        res = mgr.get_results("batch")
        assert [r.status for r in res] == ["succeeded", "succeeded"]
        assert all(isinstance(r, BatchResult) for r in res)

        # Erreurs uniquement
        mgr = self._anthropic_manager([self._error_result("a1"), self._error_result("a2")])
        res = mgr.get_results("batch")
        assert [r.status for r in res] == ["failed", "failed"]
        assert "bad request" in res[0].error["message"]

        # Mixte
        mgr = self._anthropic_manager([self._success_result("a1"), self._error_result("a2")])
        res = mgr.get_results("batch")
        assert len(res) == 2
        assert {r.status for r in res} == {"succeeded", "failed"}


# =============================================================================
# Point d'entrée pour pytest
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
