"""
Test du filtrage des paramètres pour éviter les erreurs d'API
=============================================================
Vérifie que les paramètres non supportés sont bien filtrés.
"""

from unittest.mock import MagicMock, patch

# Import du module
from ia_provider import manager
from ia_provider.openai import OpenAIProvider
from ia_provider.anthropic import AnthropicProvider


def test_openai_filters_top_k():
    """Test que top_k est filtré pour OpenAI."""
    print("Test: Filtrage de top_k pour OpenAI")
    print("-" * 40)
    
    with patch('ia_provider.openai.openai') as mock_openai:
        # Configurer le mock
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test réponse"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Créer le provider
        provider = OpenAIProvider("gpt-4.1", "test-key")
        
        # Appeler avec top_k (qui devrait être filtré)
        response = provider.generer_reponse(
            "Test prompt",
            temperature=0.5,
            max_tokens=100,
            top_k=40,  # Ce paramètre devrait être filtré
            frequency_penalty=0.5  # Ce paramètre devrait passer
        )
        
        # Vérifier les paramètres envoyés à l'API
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        
        print(f"✓ Réponse reçue: {response}")
        print(f"✓ Paramètres envoyés: {list(call_kwargs.keys())}")
        
        # Assertions
        assert 'top_k' not in call_kwargs, "top_k ne devrait pas être dans les paramètres"
        assert 'frequency_penalty' in call_kwargs, "frequency_penalty devrait être présent"
        assert call_kwargs.get('max_completion_tokens') == 100, "max_tokens devrait être converti"
        
        print("✅ Test réussi: top_k a été filtré\n")


def test_anthropic_filters_unsupported():
    """Test que les paramètres non supportés sont filtrés pour Anthropic."""
    print("Test: Filtrage des paramètres non supportés pour Anthropic")
    print("-" * 40)
    
    with patch('ia_provider.anthropic.anthropic') as mock_anthropic:
        # Configurer le mock
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test réponse Claude")]
        mock_client.messages.create.return_value = mock_response
        
        # Créer le provider
        provider = AnthropicProvider("claude-sonnet-4", "test-key")
        
        # Appeler avec des paramètres non supportés
        response = provider.generer_reponse(
            "Test prompt",
            temperature=0.5,
            max_tokens=200,
            top_k=40,  # Non supporté par Anthropic
            frequency_penalty=0.5,  # Non supporté par Anthropic
            presence_penalty=0.5,  # Non supporté par Anthropic
            thinking_budget=100  # Supporté pour Claude 4
        )
        
        # Vérifier les paramètres envoyés à l'API
        call_kwargs = mock_client.messages.create.call_args[1]
        
        print(f"✓ Réponse reçue: {response}")
        print(f"✓ Paramètres envoyés: {list(call_kwargs.keys())}")
        
        # Assertions
        assert 'top_k' not in call_kwargs, "top_k ne devrait pas être présent"
        assert 'frequency_penalty' not in call_kwargs, "frequency_penalty ne devrait pas être présent"
        assert 'presence_penalty' not in call_kwargs, "presence_penalty ne devrait pas être présent"
        assert 'thinking' in call_kwargs, "thinking devrait être présent"
        assert call_kwargs['thinking']['budget_tokens'] == 100, "thinking_budget devrait être converti"
        
        print("✅ Test réussi: paramètres non supportés filtrés\n")


def test_parameter_filtering_with_manager():
    """Test via le manager avec config par défaut contenant top_k."""
    print("Test: Filtrage via le manager")
    print("-" * 40)
    
    with patch('ia_provider.openai.openai') as mock_openai:
        # Configurer le mock
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test via manager"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Utiliser le manager (qui charge config.yaml avec top_k par défaut)
        provider = manager.get_provider("gpt-4.1", "test-key")
        
        # Générer une réponse
        response = provider.generer_reponse("Test")
        
        # Vérifier que top_k n'est pas passé même s'il est dans la config
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        
        print(f"✓ Config par défaut contient top_k: {manager.get_default_param('top_k')}")
        print(f"✓ Paramètres filtrés envoyés à l'API: {list(call_kwargs.keys())}")
        
        assert 'top_k' not in call_kwargs, "top_k de la config devrait être filtré"
        
        print("✅ Test réussi: top_k de la config par défaut est filtré\n")


def main():
    """Exécute tous les tests."""
    print("\n" + "=" * 50)
    print("TESTS DU FILTRAGE DES PARAMÈTRES")
    print("=" * 50 + "\n")
    
    try:
        test_openai_filters_top_k()
        test_anthropic_filters_unsupported()
        test_parameter_filtering_with_manager()
        
        print("=" * 50)
        print("✅ TOUS LES TESTS RÉUSSIS")
        print("=" * 50)
        print("\nLe problème 'top_k' est corrigé!")
        print("Les paramètres non supportés sont automatiquement filtrés.")
        
    except AssertionError as e:
        print(f"\n❌ ÉCHEC DU TEST: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        raise


if __name__ == "__main__":
    main()