# test_contexto.py
import unittest
from unittest.mock import patch, MagicMock, call, ANY
import numpy as np
import sys
import io
import json

from contexto import GamePlayer, COLLECTION_NAME, TEXTMODEL, RERANKER

# Epsilon value used in the benchmark method
EPSILON_BENCH = 1e-9


class TestGamePlayerStatistics(unittest.TestCase):

    def setUp(self):
        self.nltk_patcher = patch("contexto.nltk")
        self.mock_nltk = self.nltk_patcher.start()
        self.addCleanup(self.nltk_patcher.stop)

        self.wordnet_patcher = patch("contexto.wordnet")
        self.mock_wordnet = self.wordnet_patcher.start()
        self.addCleanup(self.wordnet_patcher.stop)

        self.requests_patcher = patch("contexto.requests")
        self.mock_requests = self.requests_patcher.start()
        self.addCleanup(self.requests_patcher.stop)

        self.QdrantClient_patcher = patch("contexto.QdrantClient")
        self.mock_QdrantClient_class = self.QdrantClient_patcher.start()
        self.addCleanup(self.QdrantClient_patcher.stop)

        self.qdrant_models_patcher = patch("contexto.models")
        self.mock_qdrant_models = self.qdrant_models_patcher.start()
        self.addCleanup(self.qdrant_models_patcher.stop)

        self.TextEmbedding_patcher = patch("contexto.TextEmbedding")
        self.mock_TextEmbedding_class = self.TextEmbedding_patcher.start()
        self.addCleanup(self.TextEmbedding_patcher.stop)

        self.TextCrossEncoder_patcher = patch("contexto.TextCrossEncoder")
        self.mock_TextCrossEncoder_class = self.TextCrossEncoder_patcher.start()
        self.addCleanup(self.TextCrossEncoder_patcher.stop)

        self.mock_nltk.corpus.wordnet = self.mock_wordnet
        self.mock_wordnet.ensure_loaded.return_value = None
        self.mock_wordnet.synsets.return_value = []
        self.mock_nltk.download.return_value = None

        self.mock_qdrant_client_instance = MagicMock(name="mock_qdrant_client_instance")
        self.mock_qdrant_client_instance.get_collection.return_value.points_count = 1000
        self.mock_QdrantClient_class.return_value = self.mock_qdrant_client_instance

        self.mock_embedding_model_instance = MagicMock(name="mock_embedding_model_instance")
        self.mock_TextEmbedding_class.return_value = self.mock_embedding_model_instance

        self.mock_reranker_model_instance = MagicMock(name="mock_reranker_model_instance")
        self.mock_TextCrossEncoder_class.return_value = self.mock_reranker_model_instance

        self.get_word_distance_patcher = patch("contexto.GamePlayer.get_word_distance")
        self.mock_get_word_distance = self.get_word_distance_patcher.start()
        self.addCleanup(self.get_word_distance_patcher.stop)
        self.mock_get_word_distance.return_value = 100

        self.player = GamePlayer(
            client=self.mock_qdrant_client_instance, game_number=0, starting_id=0, rerank=False
        )

        self.benchmark_game_outcomes = []
        self.benchmark_replay_call_count = 0

    def _set_active_strategy(self, strategy_name):
        strategies = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
        for s in strategies:
            setattr(self.player, s, False)
        if strategy_name:
            setattr(self.player, strategy_name, True)

    def _mock_replay_for_benchmark_side_effect(self, reset_flag_from_benchmark_call):
        current_outcome = self.benchmark_game_outcomes[self.benchmark_replay_call_count]
        won, tries, game_specific_stats = (
            current_outcome["won"],
            current_outcome["tries"],
            current_outcome["stats"],
        )

        for stat_type in ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]:
            setattr(self.player, f"{stat_type}_tries", 0)
            setattr(self.player, f"{stat_type}_error", 0.0)
            setattr(self.player, f"{stat_type}_top", 0)

        for strategy, stats_values in game_specific_stats.items():
            setattr(self.player, f"{strategy}_tries", stats_values["tries"])
            setattr(self.player, f"{strategy}_error", float(stats_values["error"]))
            setattr(self.player, f"{strategy}_top", stats_values["top"])

        self.benchmark_replay_call_count += 1
        return won, tries, []

    def test_initial_statistics_are_zero_or_default(self):
        self.assertEqual(self.player.noisy_mean_tries, 0)
        self.assertEqual(self.player.noisy_mean_error, 0)
        self.assertEqual(self.player.noisy_mean_top, 0)
        self.assertEqual(self.player.noisy_best_tries, 0)
        self.assertEqual(self.player.noisy_best_error, 0)
        self.assertEqual(self.player.noisy_best_top, 0)
        self.assertEqual(self.player.non_noisy_tries, 0)
        self.assertEqual(self.player.non_noisy_error, 0)
        self.assertEqual(self.player.non_noisy_top, 0)
        self.assertEqual(self.player.opposite_tries, 0)
        self.assertEqual(self.player.opposite_error, 0)
        self.assertEqual(self.player.opposite_top, 0)
        self.assertEqual(self.player.synonym_tries, 0)
        self.assertEqual(self.player.synonym_error, 0)
        self.assertEqual(self.player.synonym_top, 0)
        self.assertEqual(self.player.move_vec_tries, 0)
        self.assertEqual(self.player.move_vec_error, 0)
        self.assertEqual(self.player.move_vec_top, 0)
        self.assertEqual(self.player.tries, 1)
        self.assertEqual(self.player.error_run, 0)
        self.assertTrue(isinstance(self.player.results, list) and not self.player.results)

    def test_add_result_updates_stats_non_noisy_new_best(self):
        self._set_active_strategy("non_noisy")
        initial_best_dist = self.player.best_dist
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=50, id=1, vec=np.array([0.1]), word="word1")
        self.assertEqual(self.player.non_noisy_tries, 1)
        self.assertEqual(self.player.non_noisy_error, 50 - initial_best_dist)
        self.assertEqual(self.player.non_noisy_top, 1)
        self.assertEqual(self.player.tries, 2)
        self.assertEqual(self.player.error_run, 0)
        self.assertEqual(len(self.player.results), 1)
        res_dist, res_id, res_vec, res_word = self.player.results[0]
        self.assertEqual(res_dist, 50)
        self.assertEqual(res_id, 1)
        np.testing.assert_array_equal(res_vec, np.array([0.1]))
        self.assertEqual(res_word, "word1")

    def test_add_result_updates_stats_noisy_best_not_new_best(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            self._set_active_strategy("non_noisy")
            self.player.add_result(distance=20, id=1, vec=np.array([0.1]), word="best_word")
        self.player.non_noisy_tries = 0
        self.player.non_noisy_error = 0
        self.player.non_noisy_top = 0

        self._set_active_strategy("noisy_best")
        current_best_dist = self.player.best_dist
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=30, id=2, vec=np.array([0.2]), word="word2")
        self.assertEqual(self.player.noisy_best_tries, 1)
        self.assertEqual(self.player.noisy_best_error, 30 - current_best_dist)
        self.assertEqual(self.player.noisy_best_top, 0)
        self.assertEqual(self.player.tries, 3)
        initial_error_run = 0
        if 30 > self.player.priority_threshold and not self.player.synonym and not self.player.opposite:
            self.assertEqual(self.player.error_run, initial_error_run + 1)
        else:
            self.assertEqual(self.player.error_run, initial_error_run)

    def test_add_result_duplicate_word_does_not_add(self):
        with patch("sys.stdout", new_callable=io.StringIO):
            self._set_active_strategy("non_noisy")
            self.player.add_result(distance=10, id=1, vec=np.array([0.1]), word="word1")
        initial_tries = self.player.tries
        initial_results_len = len(self.player.results)
        initial_non_noisy_tries = self.player.non_noisy_tries
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            self.player.add_result(distance=10, id=1, vec=np.array([0.1]), word="word1")
        self.assertEqual(self.player.tries, initial_tries)
        self.assertEqual(len(self.player.results), initial_results_len)
        self.assertEqual(self.player.non_noisy_tries, initial_non_noisy_tries)
        self.assertIn("Tried adding duplicate result: word1", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_single_win(self, mock_stdout):
        game_stats = {
            "non_noisy": {
                "tries": 5,
                "error": 10.0,
                "top": 2,
            },  # Per-game avg error for this strategy: (10.0+e)/5 = 2.0
            "noisy_best": {
                "tries": 3,
                "error": 6.0,
                "top": 1,
            },  # Per-game avg error for this strategy: (6.0+e)/3 = 2.0
        }
        self.benchmark_game_outcomes = [{"won": True, "tries": 15, "stats": game_stats}]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(1)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 1 games played:", output)
        self.assertIn("Mean guesses per game: 15.00", output)

        # Expected calculation for non_noisy:
        # total_error_contribution_from_this_game = (game_stats["non_noisy"]["error"] + EPSILON_BENCH) / game_stats["non_noisy"]["tries"]
        # avg_error_overall = total_error_contribution_from_this_game / 1 (since 1 game where this strategy was used)
        expected_nn_err_val = (game_stats["non_noisy"]["error"] + EPSILON_BENCH) / game_stats["non_noisy"][
            "tries"
        ]
        expected_nn_top_val = float(game_stats["non_noisy"]["top"]) / 1.0  # 1 game where strategy used
        self.assertIn(f"non_noisy error average per guess per game: {expected_nn_err_val:.2f}", output)
        self.assertIn(f"non_noisy best guess average per game: {expected_nn_top_val:.2f}", output)

        expected_nb_err_val = (game_stats["noisy_best"]["error"] + EPSILON_BENCH) / game_stats["noisy_best"][
            "tries"
        ]
        expected_nb_top_val = float(game_stats["noisy_best"]["top"]) / 1.0  # 1 game where strategy used
        self.assertIn(f"noisy_best error average per guess per game: {expected_nb_err_val:.2f}", output)
        self.assertIn(f"noisy_best best guess average per game: {expected_nb_top_val:.2f}", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_multiple_games_mixed_stats_and_loss(self, mock_stdout):
        game1_stats = {
            "non_noisy": {"tries": 10, "error": 20.0, "top": 3},
            "synonym": {"tries": 2, "error": 5.0, "top": 0},
        }
        game3_stats = {
            "non_noisy": {"tries": 8, "error": 8.0, "top": 1},
            "move_vec": {"tries": 5, "error": 15.0, "top": 2},
        }
        self.benchmark_game_outcomes = [
            {"won": True, "tries": 25, "stats": game1_stats},
            {"won": False, "tries": 50, "stats": {}},
            {"won": True, "tries": 15, "stats": game3_stats},
        ]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(3)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 2 games played:", output)
        self.assertIn(f"Mean guesses per game: {(25.0+15.0)/3.0:.2f}", output)

        # Non-noisy: Used in game1 and game3
        nn_g1_avg_err = (game1_stats["non_noisy"]["error"] + EPSILON_BENCH) / game1_stats["non_noisy"][
            "tries"
        ]  # 2.0
        nn_g3_avg_err = (game3_stats["non_noisy"]["error"] + EPSILON_BENCH) / game3_stats["non_noisy"][
            "tries"
        ]  # 1.0
        expected_nn_err_overall_avg = (nn_g1_avg_err + nn_g3_avg_err) / 2.0  # (2.0 + 1.0) / 2 = 1.5
        expected_nn_top_overall_avg = (
            game1_stats["non_noisy"]["top"] + game3_stats["non_noisy"]["top"]
        ) / 2.0  # (3+1)/2 = 2.0
        self.assertIn(
            f"non_noisy error average per guess per game: {expected_nn_err_overall_avg:.2f}", output
        )
        self.assertIn(f"non_noisy best guess average per game: {expected_nn_top_overall_avg:.2f}", output)

        # Synonym: Used in game1 only
        syn_g1_avg_err = (game1_stats["synonym"]["error"] + EPSILON_BENCH) / game1_stats["synonym"][
            "tries"
        ]  # 2.5
        expected_syn_err_overall_avg = syn_g1_avg_err / 1.0  # 2.5
        expected_syn_top_overall_avg = game1_stats["synonym"]["top"] / 1.0  # 0.0
        self.assertIn(f"synonym error average per guess per game: {expected_syn_err_overall_avg:.2f}", output)
        self.assertIn(f"synonym best guess average per game: {expected_syn_top_overall_avg:.2f}", output)

        # Move_vec: Used in game3 only
        mv_g3_avg_err = (game3_stats["move_vec"]["error"] + EPSILON_BENCH) / game3_stats["move_vec"][
            "tries"
        ]  # 3.0
        expected_mv_err_overall_avg = mv_g3_avg_err / 1.0  # 3.0
        expected_mv_top_overall_avg = game3_stats["move_vec"]["top"] / 1.0  # 2.0
        self.assertIn(f"move_vec error average per guess per game: {expected_mv_err_overall_avg:.2f}", output)
        self.assertIn(f"move_vec best guess average per game: {expected_mv_top_overall_avg:.2f}", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_strategy_tried_with_zero_error(self, mock_stdout):
        game_stats = {
            "non_noisy": {"tries": 5, "error": 0.0, "top": 1},
            "noisy_best": {"tries": 3, "error": 6.0, "top": 0},
        }
        self.benchmark_game_outcomes = [{"won": True, "tries": 10, "stats": game_stats}]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(1)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 1 games played:", output)
        # Non-noisy: error = 0.0, tries = 5
        # Per-game avg error = (0.0 + EPSILON_BENCH) / 5
        expected_nn_err_val = (0.0 + EPSILON_BENCH) / 5.0
        # Overall avg error = expected_nn_err_val / 1 game
        expected_nn_top_val = 1.0 / 1.0
        self.assertIn(
            f"non_noisy error average per guess per game: {expected_nn_err_val:.2f}", output
        )  # Should be "0.00"
        self.assertIn(f"non_noisy best guess average per game: {expected_nn_top_val:.2f}", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_strategy_not_tried(self, mock_stdout):
        game_stats = {"noisy_best": {"tries": 3, "error": 6.0, "top": 1}}
        self.benchmark_game_outcomes = [{"won": True, "tries": 10, "stats": game_stats}]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(1)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 1 games played:", output)
        self.assertNotIn("non_noisy error average", output)

    def test_error_run_increment_and_reset(self):
        self.player.priority_threshold = 100
        self._set_active_strategy("non_noisy")
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=150, id=1, vec=np.array([0.1]), word="word1")
        self.assertEqual(self.player.error_run, 1, "Error run should increment on bad guess")
        self.assertEqual(self.player.best_dist, 150)
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=160, id=2, vec=np.array([0.2]), word="word2")
        self.assertEqual(self.player.error_run, 2, "Error run should increment again")
        self.assertEqual(self.player.best_dist, 150)
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=50, id=3, vec=np.array([0.3]), word="word3")
        self.assertEqual(self.player.error_run, 0, "Error run should reset on new best guess")
        self.assertEqual(self.player.best_dist, 50)
        self.player.add_result(distance=120, id=4, vec=np.array([0.4]), word="word4")
        self.assertEqual(self.player.error_run, 1)
        self._set_active_strategy("synonym")
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=130, id=5, vec=np.array([0.5]), word="word5")
        self.assertEqual(self.player.error_run, 1, "Error run should not increment for synonym strategy")

    def test_best_message_updates_top_stats_for_active_strategy(self):
        self.assertEqual(self.player.non_noisy_top, 0)
        self.assertEqual(self.player.noisy_best_top, 0)
        self._set_active_strategy("non_noisy")
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.best_message("word1", 50)
        self.assertEqual(self.player.non_noisy_top, 1)
        self.assertEqual(self.player.noisy_best_top, 0)
        self.assertEqual(self.player.error_run, 0)
        self._set_active_strategy("noisy_best")
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.best_message("word2", 30)
        self.assertEqual(self.player.non_noisy_top, 1)
        self.assertEqual(self.player.noisy_best_top, 1)
        self.assertEqual(self.player.error_run, 0)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
