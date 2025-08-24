"""
Exemples d'utilisation de GPT-5
================================
Démonstration des paramètres reasoning_effort et verbosity.
"""

import os
from dotenv import load_dotenv
from ia_provider import manager, APIError

# Charger les variables d'environnement
load_dotenv()


def exemple_gpt5_reasoning():
    """Exemple avec différents niveaux de reasoning_effort."""
    print("=" * 60)
    print("GPT-5 - Niveaux de Raisonnement")
    print("=" * 60)
    
    prompt = "Résous ce problème: Si x + 2y = 10 et 3x - y = 5, trouve x et y"
    
    # Différents niveaux de raisonnement
    levels = [
        ("minimal", "low"),    # Réponse rapide sans raisonnement
        ("low", "low"),        # Raisonnement léger
        ("medium", "medium"),  # Raisonnement équilibré
        ("high", "high")       # Raisonnement approfondi
    ]
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Clé API OpenAI non trouvée")
        return
    
    for reasoning, verbosity in levels:
        print(f"\n--- Reasoning: {reasoning}, Verbosity: {verbosity} ---")
        
        try:
            provider = manager.get_provider("gpt-5", api_key)
            
            response = provider.generer_reponse(
                prompt,
                reasoning_effort=reasoning,
                verbosity=verbosity,
                max_tokens=300
            )
            
            print(f"Réponse: {response[:200]}...")
            
            if reasoning == "minimal":
                print("→ Réponse directe sans processus de raisonnement")
            elif reasoning == "high":
                print("→ Raisonnement approfondi avec explication détaillée")
                
        except APIError as e:
            print(f"❌ Erreur: {e}")


def exemple_gpt5_famille():
    """Teste toute la famille GPT-5."""
    print("\n" + "=" * 60)
    print("GPT-5 - Famille Complète")
    print("=" * 60)
    
    models = ["gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-5-chat-latest"]
    prompt = "Génère 3 idées créatives pour une application mobile"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Clé API OpenAI non trouvée")
        return
    
    for model in models:
        print(f"\n--- {model} ---")
        
        try:
            provider = manager.get_provider(model, api_key)
            
            # Paramètres adaptés au modèle
            if model == "gpt-5-nano":
                # Version rapide
                params = {"reasoning_effort": "minimal", "verbosity": "low"}
            elif model == "gpt-5-mini":
                # Version équilibrée
                params = {"reasoning_effort": "low", "verbosity": "medium"}
            elif model == "gpt-5-chat-latest":
                # Version chat optimisée
                params = {"reasoning_effort": "minimal", "temperature": 0.7}
            else:
                # GPT-5 complet
                params = {"reasoning_effort": "medium", "verbosity": "medium"}
            
            response = provider.generer_reponse(
                prompt,
                max_tokens=200,
                **params
            )
            
            print(f"Réponse: {response[:150]}...")
            
        except APIError as e:
            print(f"❌ Erreur: {e}")


def exemple_gpt5_vs_gpt41():
    """Compare GPT-5 avec GPT-4.1."""
    print("\n" + "=" * 60)
    print("Comparaison GPT-5 vs GPT-4.1")
    print("=" * 60)
    
    prompt = "Analyse les avantages et inconvénients de l'IA"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Clé API OpenAI non trouvée")
        return
    
    # Test GPT-4.1 (approche classique)
    print("\n🔹 GPT-4.1 (température classique)")
    print("-" * 40)
    try:
        provider = manager.get_provider("gpt-4.1", api_key)
        response = provider.generer_reponse(
            prompt,
            temperature=0.7,
            max_tokens=200  # Converti en max_completion_tokens
        )
        print(f"Réponse: {response[:200]}...")
    except Exception as e:
        print(f"Erreur: {e}")
    
    # Test GPT-5 (approche raisonnement)
    print("\n🔹 GPT-5 (reasoning_effort)")
    print("-" * 40)
    try:
        provider = manager.get_provider("gpt-5", api_key)
        response = provider.generer_reponse(
            prompt,
            reasoning_effort="medium",
            verbosity="medium",
            max_tokens=200  # Pas de conversion, GPT-5 utilise max_tokens
        )
        print(f"Réponse: {response[:200]}...")
    except Exception as e:
        print(f"Erreur: {e}")
    
    print("\n💡 Différences clés:")
    print("• GPT-4.1: Utilise temperature/top_p et max_completion_tokens")
    print("• GPT-5: Utilise reasoning_effort/verbosity et max_tokens")


def main():
    """Fonction principale."""
    print("\n🚀 DÉMONSTRATION GPT-5\n")
    
    # Afficher les modèles disponibles
    models = manager.get_available_models()
    gpt5_models = [m for m in models if m.startswith("gpt-5")]
    print(f"Modèles GPT-5 disponibles: {gpt5_models}")
    
    # Exemples
    exemple_gpt5_reasoning()
    exemple_gpt5_famille()
    exemple_gpt5_vs_gpt41()
    
    print("\n" + "=" * 60)
    print("✅ Démonstration terminée")
    print("=" * 60)


if __name__ == "__main__":
    main()