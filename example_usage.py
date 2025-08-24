"""
Exemple d'utilisation du module ia_provider
===========================================
Démonstration des fonctionnalités avec gpt-4.1 et claude-sonnet-4
"""

import os
from dotenv import load_dotenv

# Import du nouveau package
from ia_provider import manager, APIError

# Charger les variables d'environnement
load_dotenv()


def exemple_gpt41():
    """Exemple avec GPT-4.1 et max_completion_tokens."""
    print("=" * 50)
    print("Exemple avec GPT-4.1 famille")
    print("=" * 50)
    
    # Tester les trois modèles de la famille 4.1
    modeles = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]
    
    for model_name in modeles:
        print(f"\n--- Test avec {model_name} ---")
        try:
            # Obtenir le provider (clé depuis .env ou explicite)
            provider = manager.get_provider(model_name)
            
            # Générer une réponse simple
            prompt = "Réponds en 1 phrase: qu'est-ce que Python?"
            response = provider.generer_reponse(
                prompt,
                temperature=0.5,
                max_tokens=50  # Sera converti en max_completion_tokens
            )
            
            print(f"Prompt: {prompt}")
            print(f"Réponse: {response}")
            
        except APIError as e:
            print(f"Erreur API: {e}")
        except Exception as e:
            print(f"Erreur: {e}")


def exemple_claude_sonnet4():
    """Exemple avec Claude Sonnet 4 et mode thinking."""
    print("=" * 50)
    print("Exemple avec Claude Sonnet 4")
    print("=" * 50)
    
    try:
        # Obtenir le provider
        provider = manager.get_provider("claude-sonnet-4-20250514")
        
        # Test sans thinking
        print("1. Sans mode thinking:")
        response = provider.generer_reponse(
            "Qu'est-ce que la récursivité?",
            temperature=0.7,
            max_tokens=200
        )
        print(f"Réponse: {response[:150]}...\n")
        
        # Test avec thinking
        print("2. Avec mode thinking (200 tokens):")
        response = provider.generer_reponse(
            "Résous: Si x + 2y = 10 et 3x - y = 5, trouve x et y",
            thinking_budget=200,
            max_tokens=300
        )
        print(f"Réponse: {response}\n")
        
    except APIError as e:
        print(f"Erreur API: {e}")
    except Exception as e:
        print(f"Erreur: {e}")


def exemple_conversation():
    """Exemple de conversation avec les deux modèles."""
    print("=" * 50)
    print("Exemple de Conversation")
    print("=" * 50)
    
    messages = [
        {"role": "user", "content": "Bonjour! Je voudrais apprendre Python."},
        {"role": "assistant", "content": "Bonjour! Excellent choix! Python est un langage idéal pour débuter. Par où souhaitez-vous commencer?"},
        {"role": "user", "content": "Quels sont les types de données de base?"}
    ]
    
    # Test avec chaque modèle
    for model_name in manager.get_available_models():
        print(f"\nConversation avec {model_name}:")
        print("-" * 30)
        
        try:
            provider = manager.get_provider(model_name)
            response = provider.chatter(messages, temperature=0.7, max_tokens=200)
            print(f"Réponse: {response[:200]}...")
            
        except Exception as e:
            print(f"Erreur: {e}")


def afficher_info_module():
    """Affiche les informations sur le module."""
    print("=" * 50)
    print("Informations sur le Module")
    print("=" * 50)
    
    # Modèles disponibles
    models = manager.get_available_models()
    print(f"Nombre de modèles: {len(models)}")
    print(f"Modèles disponibles: {', '.join(models)}")
    
    # Providers enregistrés
    providers_info = manager.get_providers_info()
    print("\nProviders et leurs modèles:")
    for provider_name, model_list in providers_info.items():
        print(f"  - {provider_name}: {', '.join(model_list)}")
    
    # Paramètres par défaut
    print("\nParamètres par défaut:")
    print(f"  - Temperature: {manager.get_default_param('temperature')}")
    print(f"  - Max tokens: {manager.get_default_param('max_tokens')}")
    print(f"  - Top P: {manager.get_default_param('top_p')}")


def main():
    """Fonction principale."""
    print("\n" + "=" * 50)
    print("DÉMONSTRATION DU MODULE IA_PROVIDER")
    print("=" * 50 + "\n")
    
    # Afficher les informations
    afficher_info_module()
    
    # Vérifier la présence des clés API
    print("\n" + "=" * 50)
    print("Vérification des clés API")
    print("=" * 50)
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    print(f"OpenAI API Key: {'✅ Configurée' if has_openai else '❌ Manquante'}")
    print(f"Anthropic API Key: {'✅ Configurée' if has_anthropic else '❌ Manquante'}")
    
    # Exécuter les exemples selon les clés disponibles
    if has_openai:
        print("\n")
        exemple_gpt41()
    else:
        print("\n⚠️ Exemple GPT-4.1 ignoré (clé API manquante)")
    
    if has_anthropic:
        print("\n")
        exemple_claude_sonnet4()
    else:
        print("\n⚠️ Exemple Claude Sonnet 4 ignoré (clé API manquante)")
    
    if has_openai or has_anthropic:
        print("\n")
        exemple_conversation()
    
    print("\n" + "=" * 50)
    print("Fin de la démonstration")
    print("=" * 50)


if __name__ == "__main__":
    main()