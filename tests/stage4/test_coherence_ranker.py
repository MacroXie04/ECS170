"""Tests for train-only compatibility pairs, frozen-vector features, and loading."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch

from local_code.stage_4_code.Coherence_Ranker import (
    PairRanker, canonical, digest, load_ranker, make_pairs, question_pair,
)
from local_code.stage_4_code.Dataset_Loader import Vocabulary


class CoherenceRankerTests(unittest.TestCase):
    def records(self):
        return [
            {"id":"train/a","text":"Why do cats sleep? They need rest."},
            {"id":"train/b","text":"Why do dogs bark? They call friends."},
            {"id":"train/c","text":"Why do cats sleep? Their paws tire."},
            {"id":"train/d","text":"How can birds sing? They know tunes."},
            {"id":"train/e","text":"This is only a statement."},
        ]

    def model_and_vocab(self):
        vocab = Vocabulary(["why","do","cats","sleep","?","they","need","rest",".",
                            "dogs","bark","call","friends","how","can","birds","sing",
                            "know","tunes","their","paws","tire"])
        matrix = torch.randn((len(vocab),5),generator=torch.Generator().manual_seed(17))
        matrix = torch.nn.functional.normalize(matrix,dim=1)
        matrix[:4].zero_()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = PairRanker(matrix,vocab.tokens)
        return model,vocab

    def fixture(self,directory):
        model,vocab = self.model_and_vocab()
        run = Path(directory)/"run"
        checkpoint = run/"checkpoints/best.pt"
        checkpoint.parent.mkdir(parents=True)
        source_name = "local_code/stage_4_code/Coherence_Ranker.py"
        root = Path(__file__).resolve().parents[2]
        vocab_sha = hashlib.sha256(b"".join(t.encode()+b"\0" for t in vocab.tokens)).hexdigest()
        config = {"architecture":"frozen_glove_pair_mlp_v1","learning_rate":.001,
                  "source_sha256":{source_name:digest(root/source_name)},
                  "data_metadata":{"vocabulary_sha256":vocab_sha}}
        (run/"config.json").write_text(json.dumps(config))
        torch.save({"model":model.state_dict(),"config_hash":canonical(config),
                    "vocab":vocab.to_dict(),"epoch":3,"validation_auc":.71},checkpoint)
        return checkpoint,config,vocab

    def test_pair_boundaries_require_real_setup_and_punchline(self):
        for text in ("A cat sleeps.", "Why do cats sleep?", "Why? They need rest.",
                     "Why do cats sleep? ! ...", "Why do cats sleep? yes"):
            self.assertIsNone(question_pair(text),text)
        left,right = question_pair("Why do cats sleep? They need rest.")
        self.assertEqual(left,["why","do","cats","sleep","?"])
        self.assertEqual(right,["they","need","rest","."])

    def test_negative_pairs_stay_inside_records_and_never_equal_positives(self):
        records = self.records()
        before = copy.deepcopy(records)
        pairs = make_pairs(records,seed=97)
        positive = {(tuple(r["left"]),tuple(r["right"])) for r in pairs if r["label"]==1}
        by_id = {row["id"]:question_pair(row["text"]) for row in records}
        self.assertEqual(len(pairs),12)
        for row in pairs:
            self.assertEqual(row["left"],by_id[row["setup_id"]][0])
            self.assertEqual(row["right"],by_id[row["punchline_id"]][1])
            self.assertNotEqual(row["setup_id"],"train/e")
            self.assertNotEqual(row["punchline_id"],"train/e")
            if row["label"] == 0:
                self.assertNotIn((tuple(row["left"]),tuple(row["right"])),positive)
                self.assertNotEqual(row["setup_id"],row["punchline_id"])
            else:
                self.assertEqual(row["setup_id"],row["punchline_id"])
        self.assertEqual(records,before)
        # The two valid alternatives for cat setup remain excluded, even when
        # the same setup is associated with two genuinely positive punchlines.
        cat_negatives = [r for r in pairs if r["label"]==0 and r["setup_id"] in ("train/a","train/c")]
        self.assertTrue(all(r["punchline_id"] not in ("train/a","train/c") for r in cat_negatives))

    def test_pair_generation_is_deterministic_and_preserves_global_random_state(self):
        before = random.getstate()
        first = make_pairs(self.records(),seed=97)
        second = make_pairs(self.records(),seed=97)
        self.assertEqual(first,second)
        self.assertEqual(random.getstate(),before)
        singleton = make_pairs([self.records()[0]])
        self.assertEqual(len(singleton),1)
        self.assertEqual(singleton[0]["label"],1)

    def test_same_opening_negative_uses_available_matching_length_candidate(self):
        # Three different punchlines with identical two-word openings and
        # equal lengths mean a distinct hard alternative always exists.
        records = [{"id":str(i),"text":text} for i,text in enumerate([
            "Why do cats sleep? They need rest.",
            "Why do dogs bark? They call friends.",
            "Why do birds sing? They know tunes.",
        ])]
        by_id = {r["id"]:question_pair(r["text"]) for r in records}
        grouped = {}
        for row in make_pairs(records,seed=123):
            grouped.setdefault(row["setup_id"],{})[row["kind"]] = row
        for setup_id,rows in grouped.items():
            hard = rows["same_opening_negative"]
            self.assertEqual(hard["left"][:2],by_id[hard["punchline_id"]][0][:2])
            self.assertNotEqual(hard["punchline_id"],rows["random_negative"]["punchline_id"])
            self.assertGreaterEqual(len(hard["right"]),.5*len(rows["positive"]["right"]))
            self.assertLessEqual(len(hard["right"]),2*len(rows["positive"]["right"]))

    def test_frozen_features_handle_empty_unknown_and_normal_pairs(self):
        model,vocab = self.model_and_vocab()
        matrix_before = model.vectors.clone()
        pairs = [{"left":[],"right":[]},
                 {"left":["unknown_word"],"right":["another_unknown"]},
                 {"left":["why","do","cats","sleep","?"],"right":["they","need","rest","."]}]
        features = model.features(pairs)
        self.assertEqual(tuple(features.shape),(3,24))
        self.assertTrue(torch.isfinite(features).all())
        self.assertTrue(torch.equal(features[0],torch.zeros(24)))
        self.assertTrue(torch.equal(features[1],torch.zeros(24)))
        self.assertEqual(tuple(model.features([]).shape),(0,24))
        self.assertNotIn("vectors",dict(model.named_parameters()))
        self.assertFalse(features.requires_grad)
        loss = model(features).sum()
        loss.backward()
        self.assertIsNone(model.vectors.grad)
        torch.testing.assert_close(model.vectors,matrix_before,rtol=0,atol=0)

    def test_scoring_preserves_mode_rng_and_invalid_positions(self):
        model,_ = self.model_and_vocab()
        model.train()
        torch_before = torch.get_rng_state().clone()
        python_before = random.getstate()
        texts = ["Why do cats sleep? They need rest.","No question here.",
                 "Why do cats sleep? They need rest.","Why do cats sleep?"]
        scores = model.score_texts(texts)
        self.assertTrue(model.training)
        self.assertEqual(scores[0],scores[2])
        self.assertIsNone(scores[1]);self.assertIsNone(scores[3])
        self.assertGreaterEqual(scores[0],0.);self.assertLessEqual(scores[0],1.)
        self.assertEqual(scores,model.score_texts(texts))
        self.assertEqual(model.score_texts([]),[])
        torch.testing.assert_close(torch.get_rng_state(),torch_before,rtol=0,atol=0)
        self.assertEqual(random.getstate(),python_before)
        model.eval()
        model.score_texts(texts)
        self.assertFalse(model.training)

    def test_loader_round_trip_preserves_rng_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint,_,vocab = self.fixture(temp)
            before = torch.get_rng_state().clone()
            model,meta = load_ranker(checkpoint)
            torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
            self.assertFalse(model.training)
            self.assertEqual(model.vocab_tokens,vocab.tokens)
            self.assertEqual(meta["checkpoint_sha256"],digest(checkpoint))
            self.assertEqual(meta["epoch"],3);self.assertEqual(meta["validation_auc"],.71)
            self.assertIsNotNone(model.score_texts(["Why do cats sleep? They need rest."])[0])

    def test_loader_rejects_config_and_vocabulary_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint,config,vocab = self.fixture(temp)
            path = checkpoint.parent.parent/"config.json"
            config["learning_rate"] = .002
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,"configuration"):
                load_ranker(checkpoint)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint,_,_ = self.fixture(temp)
            saved = torch.load(checkpoint,weights_only=True)
            saved["vocab"]["tokens"][-1],saved["vocab"]["tokens"][-2] = saved["vocab"]["tokens"][-2],saved["vocab"]["tokens"][-1]
            torch.save(saved,checkpoint)
            with self.assertRaisesRegex(ValueError,"vocabulary"):
                load_ranker(checkpoint)

    def test_loader_rejects_source_drift_and_path_escape(self):
        for escaped in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                checkpoint,config,_ = self.fixture(temp)
                name = "../../outside.py" if escaped else "local_code/stage_4_code/Coherence_Ranker.py"
                config["source_sha256"] = {name:"0"*64}
                (checkpoint.parent.parent/"config.json").write_text(json.dumps(config))
                saved = torch.load(checkpoint,weights_only=True)
                saved["config_hash"] = canonical(config)
                torch.save(saved,checkpoint)
                with self.assertRaisesRegex(ValueError,"source"):
                    load_ranker(checkpoint)


if __name__=="__main__":
    unittest.main()

