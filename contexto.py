# pip install requests numpy qdrant_client fastembed nltk
import nltk
from nltk.corpus import wordnet
import json
import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import ExtendedPointId
from qdrant_client.http.models import QueryResponse
import numpy as np
from typing import Iterable
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
import argparse
import random
import sys

try:
    wordnet.ensure_loaded()
except LookupError:
    print("Downloading WordNet resources...")
    nltk.download("wordnet")
    from nltk.corpus import wordnet

COLLECTION_NAME = "contexto"
# Qdrant client settings here
CLIENT_URL = "http://127.0.0.1"
CLIENT_PORT = 6333
CLIENT_GRPC_PORT = 6334
CLIENT_USE_GRPC = True
TEXTMODEL = "jinaai/jina-embeddings-v2-small-en"
RERANKER = "jinaai/jina-reranker-v2-base-multilingual"


def collect_words():
    # discovered API at: https://contexto.me/static/js/gameApi.js
    # this will grab all unique words from top 500 words for games 0..964
    # ...beats using thousands of unknown words from NLTK =)
    words = set()
    for x in range(965):
        print(f"Getting top 500 words for game {x}...")
        r = requests.get(f"https://api.contexto.me/machado/en/top/{x}").json()
        words.update(r["words"])
    with open("words.json", "w") as f:
        f.write(json.dumps(list(words)))
    print(f"{len(words)} words saved to words.json!")


def get_synonyms(word):
    """Get synonyms for word. Always returns a list"""
    synonyms = set()
    synsets = wordnet.synsets(word)
    if not synsets:
        return []
    for synset in synsets:
        for lemma in synset.lemmas():
            name = lemma.name()
            # skip uppercase words and words with spaces
            # _ is placeholder for spaces in the words
            if name[0].isupper() or "_" in name:
                continue
            if name != word:
                synonyms.add(name)
    return list(synonyms)


def add_noise(vec, norm=True):
    """Add Gaussian noise to a vector and optionally re-normalize it"""
    noise_level = random.uniform(0.02, 0.08)
    vec = np.asarray(vec, dtype=np.float32)
    noise = np.random.normal(0, noise_level, vec.shape)
    noisy_embedding = vec + noise
    if norm:
        original_norm = np.linalg.norm(vec)
        current_norm = np.linalg.norm(noisy_embedding)
        if current_norm > 0:
            noisy_embedding = (noisy_embedding / current_norm) * original_norm
    return noisy_embedding.astype(np.float32)


def move_vec(from_vec, to_vec):
    """nudge from_vec towards to_vec"""
    alpha = random.uniform(0.4, 0.6)
    direction = np.asarray(to_vec, dtype=np.float32) - np.asarray(
        from_vec, dtype=np.float32
    )
    return (from_vec + alpha * direction).astype(np.float32)


def init_collection(client: QdrantClient) -> int:
    """Create or check Qdrant collection and return points_count or 0 on error"""
    global COLLECTION_NAME
    with open("words.json", "r") as f:
        t = f.read()
        words = json.loads(t)
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists, skipping init.")
        print(f"{info.points_count} points in collection.")
        return info.points_count
    res = client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=client.get_embedding_size(TEXTMODEL),
                distance=models.Distance.COSINE,
                on_disk=False,
            ),
        },
        on_disk_payload=False,
    )
    if not res:
        print(f"Error creating collection.")
        return 0
    else:
        print("Collection created.")
    embedding_model = TextEmbedding(TEXTMODEL)
    print("Generating text embeddings for wordlist...")
    vecs = list(embedding_model.embed(words))
    points = [
        models.PointStruct(id=x, vector={"dense": vecs[x]}, payload={"word": words[x]})
        for x in range(len(vecs))
    ]
    print("Upserting points...")
    client.batch_update_points(
        collection_name=COLLECTION_NAME,
        update_operations=[
            models.UpsertOperation(upsert=models.PointsList(points=points))
        ],
        wait=True,
    )
    client.create_payload_index(COLLECTION_NAME, "word", "text", wait=True)
    return len(vecs)


def cosine_sim(vec1, vec2):
    """Return cosine similarity between vec1 and vec2"""
    vec1 = np.asarray(vec1, dtype=np.float32)
    vec2 = np.asarray(vec2, dtype=np.float32)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        raise ValueError(f"weird input = {vec1, vec2}")
    return (dot_product / (norm_vec1 * norm_vec2)).astype(np.float32)


def average_vecs(vecs, normalize=True):
    """Return normalized mean of vecs"""
    mean_vec = np.mean(vecs, axis=0)
    if normalize:
        norm = np.linalg.norm(mean_vec)
        if norm == 0:
            raise ValueError(f"weird input = {vecs}")
        return (mean_vec / norm).astype(np.float32)
    return mean_vec.astype(np.float32)

def get_word_vec(
    client: QdrantClient, word: str
) -> tuple[ExtendedPointId, Iterable, str] | tuple[None, None, None]:
    """Get closest Qdrant ID and vector for word

    Args:
        client (QdrantClient): client
        word (str): word

    Returns:
        tuple[ExtendedPointId, Iterable, str]: (id, vector, word) or (None, None, None)
    """
    record = client.scroll(
        COLLECTION_NAME,
        limit=1,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="word", match=models.MatchValue(value=word))]
        ),
        with_vectors=True,
        with_payload=True,
    )
    if record and record[0]:
        return record[0][0].id, record[0][0].vector["dense"], record[0][0].payload["word"]
    else:
        embedding_model = TextEmbedding(TEXTMODEL)
        vec = list(embedding_model.embed(word))[0]
        qr = client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=vec,
            with_vectors=True,
            with_payload=True,
            limit=1,
        )
        if len(qr.points) == 1:
            return (
                qr.points[0].id,
                qr.points[0].vector["dense"],
                qr.points[0].payload["word"],
            )
        return None, None, None


class GamePlayer:
    def __init__(
        self,
        client: QdrantClient,
        game_number: int | None = None,
        starting_id: ExtendedPointId | None = None,
        rerank=False,
        fast_fail=False,
        all_vecs: None | list = None,
    ):
        """Initialize GamePlayer

        Args:
            client (QdrantClient): QdrantClient to use
            game_number (int | None, optional): Game to play, if None choose random. Defaults to None.
            starting_id (int | None, optional): Word ID to play, if None choose random. Defaults to None.
            rerank (bool, optional): if True enable reranker. Defaults to False.
            fail_fast (bool, optional): if True, end strategy on really bad guess. Defaults to False.
            all_vecs (None | list, optional): if provided enable "opposite words". Defaults to None.
        """
        self.client = client
        self.game_number = (
            game_number if game_number is not None else random.randint(0, 964)
        )
        assert 0 <= self.game_number < 965
        self.fast_fail = fast_fail
        self.rerank = rerank
        if rerank:
            self.reranker = TextCrossEncoder(RERANKER)
        self.embedding_model = TextEmbedding(TEXTMODEL)
        self.points_count = client.get_collection(COLLECTION_NAME).points_count
        self.all_vecs = all_vecs
        self.starting_id = (
            starting_id if starting_id else random.randint(0, self.points_count)
        )
        # collect some stats
        self.noisy_mean_tries = 0
        self.noisy_mean_error = 0
        self.noisy_mean_top = 0
        self.noisy_best_tries = 0
        self.noisy_best_error = 0
        self.noisy_best_top = 0
        self.non_noisy_tries = 0
        self.non_noisy_error = 0
        self.non_noisy_top = 0
        self.opposite_tries = 0
        self.opposite_error = 0
        self.opposite_top = 0
        self.synonym_tries = 0
        self.synonym_error = 0
        self.synonym_top = 0
        self.move_vec_tries = 0
        self.move_vec_error = 0
        self.move_vec_top = 0
        self.tries = 1
        # last try number we tried a random word
        self.last_random = 0
        # how many times in a row we failed to improve
        self.error_run = 0
        # after this many try alternate strategy: synonyms and opposite words
        self.error_threshold = 5
        self.limit = 10
        # guess random words until below this range:
        # word guesses higher than this will count as an error
        # afer self.error_run errors, alternate strategies will be tried
        self.priority_threshold = 2_000
        # if contexto dist >= this and fast_fail is on, abandon current strategy
        self.reject_threshold = 5_000
        self.noisy_mean = False
        self.noisy_best = False
        self.non_noisy = False
        self.opposite = False
        self.synonym = False
        self.move_vec = False
        # force fast fail for the current round only
        self.force_fast_fail = False
        self.results = []
        self.tried_words = []
        self.max_tries = 10_000

    def replay(self, reset=True) -> tuple[bool, int, list[tuple[int, int, Iterable, str]]]:
        """Play another random game, for the benchmark below"""
        if reset:
            fast_fail = self.fast_fail
            rerank = self.rerank
            all_vecs = self.all_vecs
            self.__init__(
                self.client,
                rerank=rerank,
                all_vecs=all_vecs,
                fast_fail=fast_fail,
            )
        return self.play()

    def benchmark(self, number: int):
        """Keep track of game stats
        error = distance difference between current guess and best guess
        if current guess distance < best guess distance, error goes down
        Per game stats:
            non_noisy mean error = average error per guess for regular query
            non_noisy best word guesses = how many times this strategy got best guess
            etc.
        Overall stats:
            Average of the above stats per game
            Tries per game

        Args:
            number (int): number of random games to play
        """
        lowest = 40_000
        highest = 0
        total = 0
        lost = 0
        total_non_noisy_error = 0
        total_non_noisy_top = 0
        # every strategy isn't necessarily used every game
        total_non_noisy_games = 0
        total_noisy_mean_error = 0
        total_noisy_mean_top = 0
        total_noisy_mean_games = 0
        total_noisy_best_error = 0
        total_noisy_best_top = 0
        total_noisy_best_games = 0
        total_move_vec_error = 0
        total_move_vec_top = 0
        total_move_vec_games = 0
        total_synonym_error = 0
        total_synonym_top = 0
        total_synonym_games = 0
        total_opposite_error = 0
        total_opposite_top = 0
        total_opposite_games = 0
        reset = False
        
        for x in range(number):
            print(f"Playing game: {x + 1}")
            won, tries, _ = self.replay(reset)
            reset = True
            e = 1e-9
            if not won:
                lost += 1
                continue
            if tries < lowest:
                lowest = tries
            if tries > highest:
                highest = tries
            if self.non_noisy_tries:
                total_non_noisy_error += (self.non_noisy_error + e) / self.non_noisy_tries
                total_non_noisy_games += 1
                total_non_noisy_top += self.non_noisy_top
            if self.noisy_mean_tries:
                total_noisy_mean_error += (self.noisy_mean_error + e) / self.noisy_mean_tries
                total_noisy_mean_games += 1
                total_noisy_mean_top += self.noisy_mean_top
            if self.noisy_best_tries:
                total_noisy_best_error += (self.noisy_best_error + e) / self.noisy_best_tries
                total_noisy_best_games += 1
                total_noisy_best_top += self.noisy_best_top
            if self.move_vec_tries:
                total_move_vec_error += (self.move_vec_error + e) / self.move_vec_tries
                total_move_vec_games += 1
                total_move_vec_top += self.move_vec_top
            if self.synonym_tries:
                total_synonym_error += (self.synonym_error + e) / self.synonym_tries
                total_synonym_games += 1
                total_synonym_top += self.synonym_top
            if self.opposite_tries:
                total_opposite_error += (self.opposite_error + e) / self.opposite_tries
                total_opposite_games += 1
                total_opposite_top += self.opposite_top
            total += tries
        print(f"Stats for {number-lost} games played:")
        print(f"Highest guesses per game: {highest}")
        print(f"Lowest guesses per game: {lowest}")
        print(f"Mean guesses per game: {total / number:.2f}")
        print(f"Games abandoned: {lost}")
        if total_non_noisy_games:
            print(
                "non_noisy error average per guess per game:"
                f" {total_non_noisy_error / total_non_noisy_games:.2f}, non_noisy best"
                " guess average per game:"
                f" {total_non_noisy_top/ total_non_noisy_games:.2f}"
            )
        if total_noisy_mean_games:
            print(
                "noisy_mean error average per guess per game:"
                f" {total_noisy_mean_error / total_noisy_mean_games:.2f}, noisy_mean best"
                " guess average per game:"
                f" {total_noisy_mean_top/ total_noisy_mean_games:.2f}"
            )
        if total_noisy_best_games:
            print(
                "noisy_best error average per guess per game:"
                f" {total_noisy_best_error / total_noisy_best_games:.2f}, noisy_best best"
                " guess average per game:"
                f" {total_noisy_best_top/ total_noisy_best_games:.2f}"
            )
        if total_move_vec_games:
            print(
                "move_vec error average per guess per game:"
                f" {total_move_vec_error / total_move_vec_games:.2f}, move_vec best guess"
                f" average per game: {total_move_vec_top / total_move_vec_games:.2f}"
            )
        if total_synonym_games:
            print(
                "synonym error average per guess per game:"
                f" {total_synonym_error / total_synonym_games:.2f}, synonym best"
                f" guess average per game: {total_synonym_top / total_synonym_games:.2f}"
            )
        if total_opposite_games:
            print(
                "opposite error average per guess per game:"
                f" {total_opposite_error / total_opposite_games:.2f}, opposite best guess"
                f" average per game: {total_opposite_top / total_opposite_games:.2f}"
            )

    @property
    def best_vec(self):
        """vector for current best word guess"""
        return self.results[0][2]

    @property
    def best_dist(self):
        """contexto distance for current best word guess"""
        if self.results:
            return self.results[0][0]
        return 100_000

    @property
    def best_word(self):
        """current best word guess"""
        return self.results[0][3]

    def vec_from_word(self, word: str) -> Iterable:
        """Get vector for word
        First will attempt to get match from Qdrant
        If not found the embedding model will return the embeddings

        Args:
            client (QdrantClient): client
            word (str): word

        Returns:
            Iterable: vec
        """
        record = self.client.scroll(
            COLLECTION_NAME,
            limit=1,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="word", match=models.MatchValue(value=word))
                ]
            ),
            with_vectors=True,
        )
        if record and len(record) == 1:
            return record[0][0].vector["dense"]
        vec = list(self.embedding_model.embed(word))[0]
        return vec

    def word_from_vec(self, vec) -> tuple[ExtendedPointId, str] | tuple[None, None]:
        """Get Qdrant ID and word for vector
        Given a vector, return the closest word

        Args:
            vec: vector

        Returns:
            tuple[int, str]: id, word or None,None
        """
        qr = self.client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=vec,
            with_vectors=True,
            with_payload=True,
            limit=1,
        )
        if len(qr.points) == 1:
            return qr.points[0].id, qr.points[0].payload["word"]
        return None, None

    def rerank_response(
        self, qr: QueryResponse
    ) -> tuple[list[ExtendedPointId], Iterable, list[str]]:
        """Rerank a query
        This will take an existing QueryResponse from Qdrant and rerank it

        Args:
            qr (QueryResponse): The query to rerank

        Returns:
            tuple[list[ExtendedPointId], Iterable, list[str]]: IDs, vecs, words
        """
        words = [p.payload["word"] for p in qr.points]
        ids = [p.id for p in qr.points]
        vecs = [p.vector["dense"] for p in qr.points]
        assert len(words) == len(ids) == len(vecs)
        # print(f"before rerank: {words}")
        ranked = [
            (i, score)
            for i, score in enumerate(self.reranker.rerank(self.best_word, words))
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        reranked_words = [words[x[0]] for x in ranked]
        reranked_ids = [ids[x[0]] for x in ranked]
        reranked_vecs = [vecs[x[0]] for x in ranked]
        # print(f"after rerank: {reranked_words}")
        return reranked_ids, reranked_vecs, reranked_words

    def find_unlike_word(
        self, vec
    ) -> tuple[ExtendedPointId, Iterable, str] | tuple[None, None, None]:
        """A lame way to find a word unlike the input vector

        Args:
            vec: vector for a bad guess

        Returns:
            tuple[ExtendedPointId, Iterable, str]: id, vec, word
        """
        qr = self.client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=vec,
            with_vectors=True,
            with_payload=True,
            limit=1001,
        )
        index = random.randint(800, 1000)
        if len(qr.points) > index:
            return (
                qr.points[index].id,
                qr.points[index].vector["dense"],
                qr.points[index].payload["word"],
            )
        return None, None, None

    def win_message(self, word: str, won=True):
        """Display win or abandon game message
        Args:
            word (str): the secret word
            won (bool, optional): if False game was abandoned. Defaults to True.
        """
        if won:
            print(
                f"Game #{self.game_number} won in {self.tries} tries! Secret word ="
                f" {word}"
            )
        else:
            print(f"Abandoning game after {self.tries} tries!")
        if self.non_noisy_tries > 0:
            print(
                f"non_noisy mean error: {self.non_noisy_error/self.non_noisy_tries:.2f},"
                f" non_noisy best word guesses: {self.non_noisy_top}"
            )
        if self.noisy_mean_tries > 0:
            print(
                "noisy_mean mean error:"
                f" {self.noisy_mean_error/self.noisy_mean_tries:.2f}, noisy_mean best"
                f" word guesses: {self.noisy_mean_top}"
            )
        if self.noisy_best_tries > 0:
            print(
                "noisy_best mean error:"
                f" {self.noisy_best_error/self.noisy_best_tries:.2f}, noisy_best best"
                f" word guesses: {self.noisy_best_top}"
            )
        if self.opposite_tries > 0:
            print(
                f"opposite mean error: {self.opposite_error/self.opposite_tries:.2f},"
                f" opposite best word guesses: {self.opposite_top}"
            )
        if self.synonym_tries > 0:
            print(
                f"synonym mean error: {self.synonym_error/self.synonym_tries:.2f},"
                f" synonym best word guesses: {self.synonym_top}"
            )
        if self.move_vec_tries > 0:
            print(
                f"move_vec mean error: {self.move_vec_error/self.move_vec_tries:.2f},"
                f" move_vec best word guesses: {self.move_vec_top}"
            )

    def best_message(self, word: str, distance: int):
        """Display best guess message and record stats related to it
        Args:
            word (str): the current best guess
            distance (int): distance to the secret word
        """
        print(f"New best guess: {word}, distance = {distance}")
        if self.non_noisy:
            self.non_noisy_top += 1
        elif self.noisy_mean:
            self.noisy_mean_top += 1
        elif self.noisy_best:
            self.noisy_best_top += 1
        elif self.synonym:
            self.synonym_top += 1
        elif self.opposite:
            self.opposite_top += 1
        elif self.move_vec:
            self.move_vec_top += 1
        self.error_run = 0

    def info_message(self, word: str, distance: int):
        """Display info for current guess

        Args:
            word (str): current guess
            distance (int): distance to secret word
        """
        r_msg = " [reranking: on]" if self.rerank else ""
        t_msg = ""
        if self.non_noisy:
            t_msg = " [type: non_noisy]"
        elif self.noisy_mean:
            t_msg = " [type: noisy_mean]"
        elif self.noisy_best:
            t_msg = " [type: noisy_best]"
        elif self.synonym:
            t_msg = " [type: synonym]"
        elif self.move_vec:
            t_msg = " [type: move_vec]"
        elif self.opposite:
            t_msg = " [type: opposite]"
        f_msg = f" [fail_fast = {self.fast_fail or self.force_fast_fail}]"
        print(
            f"Try {self.tries}: Word guess: '{word}' ="
            f" {distance}{r_msg}{t_msg}{f_msg}, Current best guess ="
            f" {self.best_word}"
        )

    def get_word_distance(self, word: str) -> int:
        """Get Contexto score for word

        Args:
            game_number (int): game number to play
            word (str): word to check

        Returns:
            int: distance or -1 on error
        """
        try:
            if word in self.tried_words:
                print(f"Duplicate word: {word}")
                return -1
            x = requests.get(
                f"https://api.contexto.me/machado/en/game/{self.game_number}/{word}"
            ).json()
            if "error" in x:  # Unknown word
                print(f"Contexto.me website reports error for word: {word}")
                return -1
            return x["distance"]
        except Exception as e:
            print(f"Exception during website request for word '{word}': {e}")
            return -1

    def add_result(self, distance: int, id: int, vec: Iterable, word: str):
        """Add guess result to the list, sort the list, handle stats

        Args:
            distance (int): distance to the secret word
            id (int): Qdrant ID for the word
            vec (Iterable): vector associated with the word
            word (str): current guess
        """
        if word in self.tried_words:
            # this shouldn't happen
            print(f"Tried adding duplicate result: {word}")
            return
        if distance < self.best_dist:
            self.best_message(word, distance)
            
        self.update_stats(distance)
        self.tried_words.append(word)
        self.results.append((distance, id, vec, word))
        self.results.sort(key=lambda x: x[0])
        self.tries += 1
        if distance > self.priority_threshold and not self.synonym and not self.opposite:
            self.error_run += 1
        self.info_message(word, distance)
        if distance == 0:
            self.win_message(word)

    def update_stats(self, distance: int):
        """Update stats

        Args:
            distance (int): distance from secret word for current guess
        """
        if self.noisy_best:
            self.noisy_best_tries += 1
            self.noisy_best_error += distance - self.best_dist
        elif self.noisy_mean:
            self.noisy_mean_tries += 1
            self.noisy_mean_error += distance - self.best_dist
        elif self.non_noisy:
            self.non_noisy_tries += 1
            self.non_noisy_error += distance - self.best_dist
        elif self.move_vec:
            self.move_vec_tries += 1
            self.move_vec_error += distance - self.best_dist
        elif self.synonym:
            self.synonym_tries += 1
            self.synonym_error += distance - self.best_dist
        elif self.opposite:
            self.opposite_tries += 1
            self.opposite_error += distance - self.best_dist

    def try_opposite(self) -> bool:
        """Try to create guesses by getting words that are cosine dissimilar to bad guess
        This strategy is disabled by default because it doesn't seem very good
        Maybe it could help in instances where the guessing gets "stuck"??

        Returns:
            bool: True if game is won
        """
        if self.all_vecs and len(self.results) > 10:
            self.opposite = True
            # find words that are dissimilar from bottom ranked guesses and try them
            # this is not a very good strategy usually
            i = random.randint(0, 5)
            # get random vector from bottom of results
            bottom_vec = self.results[-i][2]
            min_sim = 1
            min_vec = None
            for v in self.all_vecs:
                sim = cosine_sim(bottom_vec, v)
                if sim < min_sim:
                    min_sim = sim
                    min_vec = v
            if not min_vec:
                self.opposite = False
                return False
            ids, vecs, words = self.do_query(min_vec)
            for i, w in enumerate(words):
                if w not in self.tried_words:
                    dist = self.get_word_distance(w)
                    if dist < 0:
                        continue
                    self.add_result(dist, ids[i], vecs[i], w)
                    if dist == 0:
                        self.opposite = False
                        return True
                    if (
                        self.force_fast_fail or self.fast_fail
                    ) and dist > self.reject_threshold:
                        self.opposite = False
                        return False
        self.opposite = False
        return False

    def do_query(
        self, vec: Iterable
    ) -> tuple[list[ExtendedPointId], Iterable, list[str]]:
        """Perform Qdrant query and optionally rerank the results

        Args:
            vec (Iterable): what to search for

        Returns:
            tuple[list[ExtendedPointId], Iterable, list[str]]: IDs, vecs, words
        """
        qr = self.client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=vec,
            limit=self.limit,
            with_payload=True,
            with_vectors=True,
        )
        if self.rerank:
            return self.rerank_response(qr)
        else:
            return (
                [p.id for p in qr.points],
                [p.vector["dense"] for p in qr.points],
                [p.payload["word"] for p in qr.points],
            )

    def try_synonyms(self) -> bool:
        """Try word synonyms
        After some bad guesses in a row, try synonyms of current best guess

        Returns:
            bool: True if game is won
        """
        self.synonym = True
        syns = get_synonyms(self.best_word)
        for s in syns:
            if s not in self.tried_words:
                vec = self.vec_from_word(s)
                ids, vecs, words = self.do_query(vec)
                for i, w in enumerate(words):
                    if w not in self.tried_words:
                        dist = self.get_word_distance(w)
                        if dist < 0:
                            continue
                        self.add_result(dist, ids[i], vecs[i], w)
                        if dist == 0:
                            self.synonym = False
                            return True
                        if (
                            self.force_fast_fail or self.fast_fail
                        ) and dist > self.reject_threshold:
                            self.synonym = False
                            return False
        self.synonym = False
        return False

    def try_word_shift(self) -> bool:
        """Shift random top guess closer to the best guess and try related words

        Returns:
            bool: True if game won
        """
        if len(self.results) > 3:
            # collect stats for move_vec
            self.move_vec = True
            # pick random guess in 2nd thru 4th place
            index = random.randint(1, 3)
            # shift random top guess closer to the best guess
            vec = move_vec(self.results[index][2], self.best_vec)
            ids, vecs, words = self.do_query(vec)
            for i, w in enumerate(words):
                if w not in self.tried_words:
                    dist = self.get_word_distance(w)
                    if dist < 0:
                        continue
                    self.add_result(dist, ids[i], vecs[i], w)
                    if dist == 0:
                        self.move_vec = False
                        return True
                    if (
                        self.force_fast_fail or self.fast_fail
                    ) and dist > self.reject_threshold:
                        self.move_vec = False
                        return False
        self.move_vec = False
        return False

    def get_random_word(self) -> tuple[ExtendedPointId, Iterable, str]:
        """Get random word from Qdrant

        Returns:
            tuple[ExtendedPointId, Iterable, str]: id, vec, word
        """
        choice = random.randint(0, self.points_count)
        qr = self.client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=choice,
            with_vectors=True,
            with_payload=True,
            limit=1,
        )
        return qr.points[0].id, qr.points[0].vector["dense"], qr.points[0].payload["word"]

    def play(
        self,
    ) -> tuple[bool, int, list[tuple[int, int, Iterable, str]]]:
        """Play contexto.me

        Returns:
            bool, int, list[tuple[score, id, vec, word]]: True if game won, number of tries it took, results
        """
        print(f"Starting Contexto.me game #{self.game_number}")
        # grab our starting choice
        pick1 = self.client.query_points(
            collection_name=COLLECTION_NAME,
            using="dense",
            query=self.starting_id,
            with_vectors=True,
            with_payload=True,
            limit=1,
        )
        dist1 = self.get_word_distance(pick1.points[0].payload["word"])
        if dist1 < 0:
            print(f"Contexto.me website error, stopping...")
            return False, self.tries, self.results
        self.add_result(
            dist1,
            self.starting_id,
            pick1.points[0].vector["dense"],
            pick1.points[0].payload["word"],
        )
        if dist1 == 0:
            return True, self.tries, self.results
        while True:
            self.noisy_mean = False
            self.noisy_best = False
            self.non_noisy = False
            self.opposite = False
            self.synonym = False
            self.force_fast_fail = False
            self.move_vec = False
            if self.tries > self.max_tries:
                self.win_message("", False)
                return False, self.tries, self.results
            if self.best_dist > self.priority_threshold:
                dist = 100_000
                # keep guessing random words until we get a decent guess
                while dist > self.priority_threshold:
                    id, vector, word = self.get_random_word()
                    if word not in self.tried_words:
                        dist = self.get_word_distance(word)
                        if dist < 0:
                            continue
                        self.add_result(dist, id, vector, word)
                        if dist == 0:
                            return True, self.tries, self.results
            else:
                # pick between 5 strategies:
                # 1) add Gaussian noise to best guess -or-
                # 2) take the normalized mean of (choice(2nd,3rd,4th) place guess, best guess), then add noise to that -or-
                # 3) try to move a random top guess closer to the best guess -or-
                # 4) just get the closest results to best guess and hope we get a better match
                # 5) every 25 tries or so guess a random word to help get the search "unstuck"
                #
                # if there has been a bad streak, choose from 2 additional strategies:
                # try WordNet synonyms for the top guess or try cosine dissimilar words to a bad guess
                # cosine dissimilar words will only be tried if self.all_vecs is not None
                vec = None
                if self.error_run > self.error_threshold:
                    self.force_fast_fail = True
                    # if opposite strategy is enabled, 50% chance to use it, otherwise just try synonyms
                    if self.all_vecs:
                        y = random.choice([True, False])
                        if y:
                            if self.try_opposite():
                                return True, self.tries, self.results
                        else:
                            if self.try_synonyms():
                                return True, self.tries, self.results
                    else:
                        if self.try_synonyms():
                            return True, self.tries, self.results
                    self.error_run -= 1  # don't want to get stuck here
                if self.tries - self.last_random >= 25:
                    self.force_fast_fail = True
                    self.last_random = self.tries
                    id, vec, word = self.get_random_word()
                    print(f"Random word picked: {word}")
                elif len(self.results) > 10:
                    c = random.choice([1, 2, 3, 4])
                    if c == 1:
                        index = random.randint(1, 3)
                        vec = add_noise(
                            average_vecs([self.best_vec, self.results[index][2]])
                        )
                        self.noisy_mean = True
                    elif c == 2:
                        # add noise to best vector
                        vec = add_noise(self.best_vec)
                        self.noisy_best = True
                    elif c == 3:
                        vec = self.best_vec
                        self.non_noisy = True
                    else:
                        vec = move_vec(self.results[-1][2], self.best_vec)
                        vec = add_noise(self.best_vec)
                        self.move_vec = True
                else:
                    # just add noise to best vector if we only have 10 or fewer guesses
                    vec = add_noise(self.best_vec)
                    self.noisy_best = True
                # get top results closest to our new vector
                ids, vecs, words = self.do_query(vec)
                for i, w in enumerate(words):
                    if w not in self.tried_words:
                        dist = self.get_word_distance(w)
                        if dist < 0:
                            continue
                        self.add_result(dist, ids[i], vecs[i], w)
                        if dist == 0:
                            return True, self.tries, self.results
                        if (
                            self.force_fast_fail or self.fast_fail
                        ) and dist > self.reject_threshold:
                            break


def word_from_vec(
    client: QdrantClient, vec
) -> tuple[ExtendedPointId, str] | tuple[None, None]:
    """Get word closest to vec

    Args:
        client (QdrantClient): client to use
        vec (_type_): vec to search for

    Returns:
        tuple[ExtendedPointId, str] | tuple[None, None]: id, word
    """
    qr = client.query_points(
        collection_name=COLLECTION_NAME,
        using="dense",
        query=vec,
        with_vectors=False,
        with_payload=True,
        limit=1,
    )
    if len(qr.points) == 1:
        return qr.points[0].id, qr.points[0].payload["word"]
    return None, None


def move_word_test(
    client: QdrantClient, from_word: str, to_word: str
) -> tuple[str | None, Iterable]:
    """Test shifting a word vector closer to another word
    This will move the vector for from_word closer to the vector for to_word
    Then a query will be made on the original vector and the new vector
    Top 10 results for each query will be shown

    Args:
        client (QdrantClient): client
        from_word (str): word to start from
        to_word (str): word to move closer to

    Returns:
        tuple[str | None, Iterable]: new word, new vec
    """
    id1, vec1, word1 = get_word_vec(client, from_word)
    id2, vec2, word2 = get_word_vec(client, to_word)
    new_vec = move_vec(vec1, vec2)
    new_id, new_word = word_from_vec(client, new_vec)
    before = client.query_points(
        collection_name=COLLECTION_NAME,
        using="dense",
        query=vec1,
        with_vectors=False,
        with_payload=True,
        limit=10,
    )
    after = client.query_points(
        collection_name=COLLECTION_NAME,
        using="dense",
        query=new_vec,
        with_vectors=False,
        with_payload=True,
        limit=10,
    )

    print(f"top 10 results closest to {word1}:")
    print(f"{', '.join(p.payload["word"] for p in before.points)}")
    print("------------------------------------------------------")
    print("top 10 results closest to new vector:")
    print(f"{', '.join(p.payload["word"] for p in after.points)}")
    return new_word, new_vec


def main(test, init, download, bench, game, word, fail, rerank, opposite):
    """Where it all starts

    Args:
        test (bool): if True, run move_word_test and quit
        init (bool): if True, delete Qdrant collection and rebuilt it
        download (bool): if True, redownload wordlist from contexto.me
        bench (int): number of random games to benchmark
        game (int): first game to play, 0-964
        word (str): first word to guess
        fail (bool): if True, enable fast_fail, which will make strategies end faster
        rerank (bool): if True, enable reranking model
        opposite (bool): if True, build and use vector cache to find cosine dissimilar words
    """
    client = QdrantClient(
        url=CLIENT_URL,
        port=CLIENT_PORT,
        grpc_port=CLIENT_GRPC_PORT,
        prefer_grpc=CLIENT_USE_GRPC,
    )
    if download:
        init = True
        collect_words()
    if init:
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
    points = init_collection(client)
    if not points:
        print("Collection init failed, exiting...")
        sys.exit(1)
    if test:
        move_word_test(client, test[0], test[1])
        sys.exit(0)
    word_id = None
    if word:
        word = word.lower()
        word_id, vec, w = get_word_vec(client, word)
        if w and w != word:
            print(f"Starting word {word} not found, using closest match: '{w}'")
        elif not w:
            print("Error selecting word to start with, exiting...")
            sys.exit(1)
    all_vecs = None
    if opposite:
        all_vecs = []
        print("Caching vectors for opposite strategy, this may take a moment...")
        try:
            records = client.scroll(
                COLLECTION_NAME, limit=points, with_vectors=True, timeout=10
            )
            all_vecs = [r.vector["dense"] for r in records[0]]
        except Exception as e:
            print(f"Exception: {e}")
            sys.exit(1)

    gp = GamePlayer(client, game, word_id, rerank, fail, all_vecs)
    gp.benchmark(bench)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contexto.me game player")
    parser.add_argument(
        "-b",
        "--bench",
        type=int,
        default=1,
        help="Number of games to benchmark (default: 1).",
    )
    parser.add_argument(
        "-g",
        "--game",
        type=int,
        default=None,
        help="Game number to play (0-964) (default: random).",
    )
    parser.add_argument(
        "-w",
        "--word",
        type=str,
        default=None,
        help="First word to guess (default: random).",
    )
    parser.add_argument(
        "-f",
        "--fail",
        action="store_true",
        help="Enable fast fail: end current strategy on really bad guess (default: False).",
    )
    parser.add_argument(
        "-r", "--rerank", action="store_true", help="Use reranker model (default: False)."
    )
    parser.add_argument(
        "-o",
        "--opposite",
        action="store_true",
        help="Use opposite word strategy (default: False).",
    )
    parser.add_argument(
        "-i",
        "--init",
        action="store_true",
        help="Delete and re-initialize Qdrant collection (default: False).",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help=(
            "Re-download wordlist and re-initialize Qdrant collection (default: False)."
        ),
    )
    parser.add_argument(
        "-t",
        "--test",
        type=str,
        default=None,
        nargs=2,
        help=(
            "Test shifting the first word closer to the second and print the results"
            " (default: None)."
        ),
    )
    args = parser.parse_args()

    main(
        args.test,
        args.init,
        args.download,
        args.bench,
        args.game,
        args.word,
        args.fail,
        args.rerank,
        args.opposite,
    )
