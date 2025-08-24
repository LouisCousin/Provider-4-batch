"""
Script de diagnostic - Vérification de la correction du bug top_k
================================================================
Ce script vérifie que le problème avec top_k est bien résolu.
"""

import os
from dotenv import load_dotenv
from ia_provider import manager, APIError

# Charger les variables d'environnement
load_dotenv()


def diagnostic_gpt41():
    """Test de diagnostic pour GPT-4.1 avec top_k."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Test de GPT-4.1 avec paramètres problématiques")
    print("=" * 60)
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⚠️  Clé API OpenAI non trouvée. Test ignoré.")
        print("   Configurez OPENAI_API_KEY dans .env pour tester.")
        return False
    
    try:
        print("\n1. Test avec top_k dans les paramètres...")
        provider = manager.get_provider("gpt-4.1", api_key)
        
        # Test qui causait l'erreur avant la correction
        response = provider.generer_reponse(
            "Dis juste 'OK' si tu reçois ce message",
            temperature=0.5,
            max_tokens=10,
            top_k=40,  # Paramètre problématique
            frequency_penalty=0,
            presence_penalty=0
        )
        
        print(f"   ✅ Succès! Réponse: {response}")
        print("   → Le paramètre top_k a été correctement filtré")
        
        # Test 2: Vérifier que les autres paramètres passent bien
        print("\n2. Test avec paramètres valides...")
        response2 = provider.generer_reponse(
            "Réponds 'TEST' en majuscules",
            temperature=0.1,
            max_tokens=10,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )
        
        print(f"   ✅ Succès! Réponse: {response2}")
        print("   → Les paramètres supportés fonctionnent correctement")
        
        return True
        
    except APIError as e:
        if "top_k" in str(e):
            print(f"   ❌ ÉCHEC: L'erreur top_k persiste!")
            print(f"   Erreur: {e}")
            print("\n   ⚠️  Le filtrage des paramètres ne fonctionne pas correctement.")
            return False
        else:
            print(f"   ❌ Erreur API différente: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur inattendue: {e}")
        return False


def diagnostic_claude():
    """Test de diagnostic pour Claude Sonnet 4."""
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: Test de Claude Sonnet 4 avec paramètres")
    print("=" * 60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("⚠️  Clé API Anthropic non trouvée. Test ignoré.")
        print("   Configurez ANTHROPIC_API_KEY dans .env pour tester.")
        return False
    
    try:
        print("\n1. Test avec paramètres non supportés...")
        provider = manager.get_provider("claude-sonnet-4", api_key)
        
        # Test avec paramètres qui devraient être filtrés
        response = provider.generer_reponse(
            "Réponds simplement 'OK'",
            temperature=0.5,
            max_tokens=10,
            top_k=40,  # Non supporté
            frequency_penalty=0.5,  # Non supporté
            presence_penalty=0.5  # Non supporté
        )
        
        print(f"   ✅ Succès! Réponse: {response}")
        print("   → Les paramètres non supportés ont été filtrés")
        
        # Test 2: Mode thinking
        print("\n2. Test du mode thinking...")
        response2 = provider.generer_reponse(
            "Calcule 2+2 et explique",
            thinking_budget=100,
            max_tokens=50
        )
        
        print(f"   ✅ Succès! Réponse: {response2[:50]}...")
        print("   → Le mode thinking fonctionne")
        
        return True
        
    except APIError as e:
        print(f"   ❌ Erreur API: {e}")
        if any(param in str(e) for param in ['top_k', 'frequency_penalty', 'presence_penalty']):
            print("   ⚠️  Des paramètres non supportés n'ont pas été filtrés!")
        return False
            
    except Exception as e:
        print(f"   ❌ Erreur inattendue: {e}")
        return False


def main():
    """Fonction principale de diagnostic."""
    print("\n" + "🔍 " * 20)
    print("DIAGNOSTIC DU MODULE IA_PROVIDER")
    print("Vérification de la correction du bug 'top_k'")
    print("🔍 " * 20)
    
    # Vérifier les modèles disponibles
    print("\n📋 Modèles disponibles:")
    models = manager.get_available_models()
    for model in models:
        print(f"   • {model}")
    
    # Tests de diagnostic
    results = []
    
    # Test GPT-4.1
    gpt_ok = diagnostic_gpt41()
    results.append(("GPT-4.1", gpt_ok))
    
    # Test Claude
    claude_ok = diagnostic_claude()
    results.append(("Claude Sonnet 4", claude_ok))
    
    # Résumé
    print("\n" + "=" * 60)
    print("RÉSUMÉ DU DIAGNOSTIC")
    print("=" * 60)
    
    for model, status in results:
        if status is None:
            continue
        emoji = "✅" if status else "❌"
        print(f"{emoji} {model}: {'OK' if status else 'PROBLÈME DÉTECTÉ'}")
    
    # Conclusion
    all_ok = all(status != False for _, status in results)
    
    if all_ok:
        print("\n🎉 SUCCÈS: Le problème 'top_k' est corrigé!")
        print("   Les paramètres non supportés sont bien filtrés.")
    else:
        print("\n⚠️  ATTENTION: Des problèmes ont été détectés.")
        print("   Vérifiez l'implémentation du filtrage des paramètres.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()