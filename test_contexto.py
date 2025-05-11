# python -m unittest
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
        self.mock_get_word_distance.return_value = 100 # Default mock return

        # Player is initialized here for tests that don't need a fresh instance per sub-test.
        # Tests that modify player state heavily or test different init paths should create their own instances.
        self.player = GamePlayer(
            client=self.mock_qdrant_client_instance, game_number=0, starting_id=0, rerank=False
        )

        self.benchmark_game_outcomes = []
        self.benchmark_replay_call_count = 0
        
        self.strategies_map = {
            "non_noisy": ("non_noisy_tries", "non_noisy_error", "non_noisy_top"),
            "noisy_mean": ("noisy_mean_tries", "noisy_mean_error", "noisy_mean_top"),
            "noisy_best": ("noisy_best_tries", "noisy_best_error", "noisy_best_top"),
            "synonym": ("synonym_tries", "synonym_error", "synonym_top"),
            "opposite": ("opposite_tries", "opposite_error", "opposite_top"),
            "move_vec": ("move_vec_tries", "move_vec_error", "move_vec_top"),
        }
        self.all_top_attrs = [attrs[2] for attrs in self.strategies_map.values()]


    def _set_active_strategy(self, strategy_name):
        # Operates on self.player by default, or a passed player instance
        player_instance = self.player 
        strategies_flags = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
        for s_flag in strategies_flags:
            setattr(player_instance, s_flag, False)
        if strategy_name:
            setattr(player_instance, strategy_name, True)

    def _mock_replay_for_benchmark_side_effect(self, reset_flag_from_benchmark_call):
        current_outcome = self.benchmark_game_outcomes[self.benchmark_replay_call_count]
        won, tries, game_specific_stats = (
            current_outcome["won"],
            current_outcome["tries"],
            current_outcome["stats"],
        )

        # Resetting stats on the self.player instance, which benchmark accesses
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

    def test_all_strategies_update_correct_top_counter_on_best_message(self):
        for strategy_flag, (_, _, top_attr_name) in self.strategies_map.items():
            with self.subTest(strategy=strategy_flag):
                # Use a fresh player for each subtest to ensure isolation
                player = GamePlayer(client=self.mock_qdrant_client_instance, game_number=0, starting_id=0)
                
                # Manually set all top counters to 0 for this player instance (already true for fresh instance)
                for attr in self.all_top_attrs:
                    self.assertEqual(getattr(player, attr), 0, f"Initial {attr} for fresh player.")
                player.error_run = 5 # Set to non-zero to check reset

                # Set the active strategy by modifying the flags on the player instance
                strategies_flags = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
                for s_flag in strategies_flags:
                    setattr(player, s_flag, False)
                setattr(player, strategy_flag, True)


                with patch("sys.stdout", new_callable=io.StringIO):
                    player.best_message("test_word", 10)

                self.assertEqual(getattr(player, top_attr_name), 1, f"{top_attr_name} should be 1 for {strategy_flag}")

                for other_top_attr in self.all_top_attrs:
                    if other_top_attr != top_attr_name:
                        self.assertEqual(getattr(player, other_top_attr), 0,
                                         f"{other_top_attr} should be 0 when {strategy_flag} is active")
                self.assertEqual(player.error_run, 0, "error_run should be reset by best_message")

    def test_add_result_new_best_updates_correct_top_and_stats(self):
        for strategy_flag, (tries_attr, error_attr, top_attr) in self.strategies_map.items():
            with self.subTest(strategy=strategy_flag):
                player = GamePlayer(
                    client=self.mock_qdrant_client_instance, game_number=0, starting_id=0
                )
                # Set active strategy on this new player instance
                strategies_flags = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
                for s_flag in strategies_flags: setattr(player, s_flag, False)
                setattr(player, strategy_flag, True)


                initial_best_dist_val = player.best_dist # 100_000 for a new player
                new_distance = 50
                
                self.assertEqual(getattr(player, top_attr), 0, f"Initial {top_attr} for {strategy_flag} should be 0")

                with patch("sys.stdout", new_callable=io.StringIO):
                    player.add_result(distance=new_distance, id=1, vec=np.array([0.1]), word=f"word_{strategy_flag}")

                self.assertEqual(getattr(player, tries_attr), 1, f"{tries_attr} should be 1 for {strategy_flag}")
                # update_stats uses best_dist *before* results are updated for the current word.
                self.assertEqual(getattr(player, error_attr), new_distance - initial_best_dist_val, f"{error_attr} calculation for {strategy_flag}")
                self.assertEqual(getattr(player, top_attr), 1, f"{top_attr} should be 1 for {strategy_flag} (new best)")
                
                for other_strat_flag, (o_tries, o_err, o_top) in self.strategies_map.items():
                    if other_strat_flag != strategy_flag:
                        self.assertEqual(getattr(player, o_tries), 0, f"Other strategy {o_tries} should be 0")
                        self.assertEqual(getattr(player, o_err), 0, f"Other strategy {o_err} should be 0")
                        self.assertEqual(getattr(player, o_top), 0, f"Other strategy {o_top} should be 0")

                self.assertEqual(player.tries, 2) 
                self.assertEqual(player.error_run, 0) 
                self.assertEqual(player.best_dist, new_distance)
                self.assertEqual(len(player.results), 1)

    def test_add_result_not_new_best_updates_correct_stats(self):
        for strategy_flag, (tries_attr, error_attr, top_attr) in self.strategies_map.items():
            with self.subTest(strategy=strategy_flag):
                player = GamePlayer(
                    client=self.mock_qdrant_client_instance, game_number=0, starting_id=0
                )
                # Set active strategy on this new player instance
                strategies_flags_list = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
                for s_flag in strategies_flags_list: setattr(player, s_flag, False)
                setattr(player, strategy_flag, True)

                established_best_distance = 20
                player.results = [(established_best_distance, 0, np.array([0.0]), "initial_best_word")]
                # player.best_dist property will now reflect established_best_distance
                
                current_best_dist_before_call = player.best_dist
                self.assertEqual(current_best_dist_before_call, established_best_distance)

                worse_distance = 30 
                initial_error_run_val = player.error_run # Should be 0 for a new player
                
                self.assertEqual(getattr(player, top_attr), 0, f"Initial {top_attr} for {strategy_flag} should be 0")

                with patch("sys.stdout", new_callable=io.StringIO):
                    player.add_result(distance=worse_distance, id=2, vec=np.array([0.2]), word=f"worse_word_{strategy_flag}")

                self.assertEqual(getattr(player, tries_attr), 1, f"{tries_attr} should be 1 for {strategy_flag}")
                self.assertEqual(getattr(player, error_attr), worse_distance - current_best_dist_before_call, f"{error_attr} calculation for {strategy_flag}")
                self.assertEqual(getattr(player, top_attr), 0, f"{top_attr} should be 0 for {strategy_flag} (not new best)")

                for other_strat_flag, (o_tries, o_err, o_top) in self.strategies_map.items():
                    if other_strat_flag != strategy_flag:
                        self.assertEqual(getattr(player, o_tries), 0, f"Other strategy {o_tries} should be 0")
                        self.assertEqual(getattr(player, o_err), 0, f"Other strategy {o_err} should be 0")
                        self.assertEqual(getattr(player, o_top), 0, f"Other strategy {o_top} should be 0")
                
                self.assertEqual(player.tries, 2) # Initial implicit 1 + 1 add_result
                # Check error_run logic:
                # if distance > priority_threshold (default 2000) AND not (synonym or opposite) then error_run++
                # Here, worse_distance (30) is not > priority_threshold (2000)
                # So, error_run should not increment due to this condition.
                # And since it wasn't a new best, best_message didn't reset it.
                self.assertEqual(player.error_run, initial_error_run_val, f"error_run for {strategy_flag}")
                
                self.assertEqual(player.best_dist, established_best_distance, "Best distance should not change")
                self.assertEqual(len(player.results), 2)


    def test_add_result_duplicate_word_does_not_add(self):
        # Use self.player as it's a sequence of calls on the same player
        self._set_active_strategy("non_noisy")
        with patch("sys.stdout", new_callable=io.StringIO):
            self.player.add_result(distance=10, id=1, vec=np.array([0.1]), word="word1")
        
        initial_tries = self.player.tries
        initial_results_len = len(self.player.results)
        initial_non_noisy_tries = self.player.non_noisy_tries
        initial_non_noisy_top = self.player.non_noisy_top # word1 was new best

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            # Attempt to add the same word again
            self.player.add_result(distance=10, id=1, vec=np.array([0.1]), word="word1")
        
        self.assertEqual(self.player.tries, initial_tries, "Tries should not change for duplicate")
        self.assertEqual(len(self.player.results), initial_results_len, "Results length should not change")
        self.assertEqual(self.player.non_noisy_tries, initial_non_noisy_tries, "Strategy tries should not change")
        self.assertEqual(self.player.non_noisy_top, initial_non_noisy_top, "Strategy top should not change")
        self.assertIn("Tried adding duplicate result: word1", mock_stdout.getvalue())

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_single_win(self, mock_stdout):
        game_stats = {
            "non_noisy": {"tries": 5, "error": 10.0, "top": 2},
            "noisy_best": {"tries": 3, "error": 6.0, "top": 1},
        }
        self.benchmark_game_outcomes = [{"won": True, "tries": 15, "stats": game_stats}]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(1)
        self.player.replay = original_replay # Restore original method
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 1 games played:", output)
        self.assertIn("Mean guesses per game: 15.00", output)
        expected_nn_err_val = (game_stats["non_noisy"]["error"] + EPSILON_BENCH) / game_stats["non_noisy"]["tries"]
        expected_nn_top_val = float(game_stats["non_noisy"]["top"]) / 1.0
        self.assertIn(f"non_noisy error average per guess per game: {expected_nn_err_val:.2f}", output)
        self.assertIn(f"non_noisy best guess average per game: {expected_nn_top_val:.2f}", output)
        expected_nb_err_val = (game_stats["noisy_best"]["error"] + EPSILON_BENCH) / game_stats["noisy_best"]["tries"]
        expected_nb_top_val = float(game_stats["noisy_best"]["top"]) / 1.0
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
            {"won": False, "tries": 50, "stats": {}}, # Lost game, stats won't be used for averages if tries=0 for a strategy
            {"won": True, "tries": 15, "stats": game3_stats},
        ]
        self.benchmark_replay_call_count = 0

        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(3)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()

        self.assertIn("Stats for 2 games played:", output) # 3 total, 1 lost
        self.assertIn(f"Mean guesses per game: {(25.0+15.0)/3.0:.2f}", output) # Mean based on total games attempted

        nn_g1_avg_err = (game1_stats["non_noisy"]["error"] + EPSILON_BENCH) / game1_stats["non_noisy"]["tries"]
        nn_g3_avg_err = (game3_stats["non_noisy"]["error"] + EPSILON_BENCH) / game3_stats["non_noisy"]["tries"]
        expected_nn_err_overall_avg = (nn_g1_avg_err + nn_g3_avg_err) / 2.0 
        expected_nn_top_overall_avg = (game1_stats["non_noisy"]["top"] + game3_stats["non_noisy"]["top"]) / 2.0
        self.assertIn(f"non_noisy error average per guess per game: {expected_nn_err_overall_avg:.2f}", output)
        self.assertIn(f"non_noisy best guess average per game: {expected_nn_top_overall_avg:.2f}", output)

        syn_g1_avg_err = (game1_stats["synonym"]["error"] + EPSILON_BENCH) / game1_stats["synonym"]["tries"]
        expected_syn_err_overall_avg = syn_g1_avg_err / 1.0
        expected_syn_top_overall_avg = game1_stats["synonym"]["top"] / 1.0
        self.assertIn(f"synonym error average per guess per game: {expected_syn_err_overall_avg:.2f}", output)
        self.assertIn(f"synonym best guess average per game: {expected_syn_top_overall_avg:.2f}", output)

        mv_g3_avg_err = (game3_stats["move_vec"]["error"] + EPSILON_BENCH) / game3_stats["move_vec"]["tries"]
        expected_mv_err_overall_avg = mv_g3_avg_err / 1.0
        expected_mv_top_overall_avg = game3_stats["move_vec"]["top"] / 1.0
        self.assertIn(f"move_vec error average per guess per game: {expected_mv_err_overall_avg:.2f}", output)
        self.assertIn(f"move_vec best guess average per game: {expected_mv_top_overall_avg:.2f}", output)

    @patch("sys.stdout", new_callable=io.StringIO)
    def test_benchmark_strategy_tried_with_zero_error(self, mock_stdout):
        game_stats = { "non_noisy": {"tries": 5, "error": 0.0, "top": 1} }
        self.benchmark_game_outcomes = [{"won": True, "tries": 10, "stats": game_stats}]
        self.benchmark_replay_call_count = 0
        original_replay = self.player.replay
        self.player.replay = MagicMock(side_effect=self._mock_replay_for_benchmark_side_effect)
        self.player.benchmark(1)
        self.player.replay = original_replay
        output = mock_stdout.getvalue()
        expected_nn_err_val = (0.0 + EPSILON_BENCH) / 5.0
        self.assertIn(f"non_noisy error average per guess per game: {expected_nn_err_val:.2f}", output)

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
        self.assertNotIn("non_noisy error average", output) # non_noisy was not in game_stats

    def test_error_run_increment_and_reset(self):
        # Use a fresh player for this specific test sequence
        player = GamePlayer(client=self.mock_qdrant_client_instance, game_number=0, starting_id=0)
        player.priority_threshold = 100 
        
        # Set active strategy (non_noisy for this test, could be any non-special one for error_run)
        strategies_flags = ["noisy_mean", "noisy_best", "non_noisy", "opposite", "synonym", "move_vec"]
        for s_flag in strategies_flags: setattr(player, s_flag, False)
        setattr(player, "non_noisy", True)


        with patch("sys.stdout", new_callable=io.StringIO): # Suppress print
            player.add_result(distance=150, id=1, vec=np.array([0.1]), word="word1") # 150 > 100 (priority_threshold)
        self.assertEqual(player.error_run, 1, "Error run should increment on bad guess above threshold")
        self.assertEqual(player.best_dist, 150)

        with patch("sys.stdout", new_callable=io.StringIO):
            player.add_result(distance=160, id=2, vec=np.array([0.2]), word="word2") # 160 > 100
        self.assertEqual(player.error_run, 2, "Error run should increment again")
        self.assertEqual(player.best_dist, 150) # best_dist unchanged

        with patch("sys.stdout", new_callable=io.StringIO):
            player.add_result(distance=50, id=3, vec=np.array([0.3]), word="word3") # New best (50 < 150)
        self.assertEqual(player.error_run, 0, "Error run should reset on new best guess")
        self.assertEqual(player.best_dist, 50)

        with patch("sys.stdout", new_callable=io.StringIO):
            player.add_result(distance=120, id=4, vec=np.array([0.4]), word="word4") # 120 > 100
        self.assertEqual(player.error_run, 1, "Error run increments again")
        
        # Test with synonym strategy (should not increment error_run based on priority_threshold)
        setattr(player, "non_noisy", False)
        setattr(player, "synonym", True)
        with patch("sys.stdout", new_callable=io.StringIO):
            player.add_result(distance=130, id=5, vec=np.array([0.5]), word="word5") # 130 > 100, but synonym active
        self.assertEqual(player.error_run, 1, "Error run should not increment for synonym strategy even if above threshold")

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)