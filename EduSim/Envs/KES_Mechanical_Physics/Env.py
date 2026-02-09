# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# -*- coding:utf-8 _*-
import pickle
import time
import os
from pathlib import Path
import networkx as nx
import numpy as np
import torch
from EduSim.Envs.KES_Mechanical_Physics.meta.Learner import KESPhysicsLeanerGroup
from EduSim.spaces import ListSpace
from EduSim.Envs.meta import Item
from EduSim.Envs.meta import Env
from EduSim.Envs.KES_Mechanical_Physics.meta.Learner import Learner
from .BuildKnowledgeStructure import BuildKG_LLM
import json
import math
from scripts.KT import Agent_DKT


from .meta import KESASSISTScorer
__all__ = ["KESPhysicsEnv"]


class KESPhysicsEnv(Env):
    def __init__(self):
        super().__init__()
        self.type = 'KES'
        self.env_name = 'KESPhysicsEnv'
        self.random_state = np.random.RandomState(2024)
        _base = Path(__file__).resolve().parent / "meta_data"
        self.dataRec_path = str(_base / "data_rec.txt")
        _p_info_path = _base / "problem_info.json"
        self.p_info = json.load(open(_p_info_path, "r", encoding="utf-8")) if _p_info_path.exists() else {}
        self.c2id_path = str(_base / "concept2id.json")
        self.KG_file = str(_base / "KG_structure.json")
        self.knowledge_structure = None
        self.num_skills = 199
        self.learning_item_base = None
        self.conceptids = self.get_concepts_id()

        self.KTnet = Agent_DKT("Mechanical_Physics")

        self.scorer = KESASSISTScorer()
        self.learners = KESPhysicsLeanerGroup(self.dataRec_path)
        self._learner = None
        self._initial_score = None
        self.episode_start_time = time.time()
        self.episode_end_time = time.time()
        self.mastery_thresh_hold = 0.5

    def get_concepts_id(self):
        concepts = {}
        with open(self.c2id_path, 'r') as f:
            for line in f:
                    key, value = line.strip().split(',')
                    concepts[key.strip('"')] = int(value)
        return concepts

    def reset(self):
        self._learner = None


    def learn_and_test(self, learner: Learner, item_id, cm=None, cb=None):
        state = learner.state
        score = self.scorer.response_function(state, item_id, cm)
        learner.learn(item_id, score)
        logs = self.update_learner_state(cb)
        return logs

    def _exam(self, learner: Learner, detailed=False, reduce="sum") -> (dict, int, float):
        if learner is None:
            return {} if detailed else 0
        state = learner.state
        knowledge_response = {}  # dict
        for test_item in learner.target:
            knowledge_response[test_item] = [test_item, self.scorer.response_function(state, test_item)]
        if detailed:
            return_thing = knowledge_response
        elif reduce == "sum":
            return_thing = np.sum([v for _, v in knowledge_response.values()])  # np.sum   []:list   knowledge_response
        elif reduce in {"mean", "ave"}:
            return_thing = np.average([v for _, v in knowledge_response.values()])
        else:
            raise TypeError("unknown reduce type %s" % reduce)  # unknown reduce type
        return return_thing

    def update_learner_state(self,cb=None):
        logs = self._learner.profile['logs']
        ans = logs[2]
        ques_text = logs[1]
        if cb:
            new_p_id = int(list(self.p_info.keys())[-1]) + 1
            logs[0][-1] = new_p_id
            self.p_info[str(new_p_id)] = {'concepts': cb, 'content': ques_text[-1]}
        kn_emb = []
        for i in logs[0]:
            ks = self.p_info[str(i)]['concepts']
            e_k = [self.conceptids[k] for k in ks]
            input_knowedge_emb = [0.] * self.num_skills
            for k in e_k:
                input_knowedge_emb[k] = 1
            kn_emb.append(input_knowedge_emb)

        msk = [1] * len(logs[0])

        self._learner._state = self.KTnet.forward_state(ans, ques_text, kn_emb, msk)

        return logs


    def begin_episode(self, *args, **kwargs):
        self._learner = next(self.learners)
        self.update_learner_state()
        self._initial_score = self._exam(self._learner)
        while self._initial_score >= len(self._learner.target):
            self._learner = next(self.learners)
            self.update_learner_state()
            self._initial_score = self._exam(self._learner)
        return self._learner.profile, self._exam(self._learner, detailed=True)


    def end_episode(self, *args, **kwargs):
        observation = self._exam(self._learner, detailed=True)
        initial_score, self._initial_score = self._initial_score, None
        final_score = self._exam(self._learner)
        reward = episode_reward(initial_score, final_score, len(self._learner.target))
        done = final_score == len(self._learner.target)
        info = {"initial_score": initial_score, "final_score": final_score}
        self.episode_end_time = time.time()
        # print('episode_env_time:' + str(self.episode_end_time - self.episode_start_time))
        return observation, reward, done, info


    def get_knowledge_structure(self):

        with open(self.KG_file, 'r') as f:
            KG = json.load(f)

        self.knowledge_structure = KG

        return KG


    def step_llm(self,ques_H, ans_H, practice_item_text,cb_mastery, cb, target, *args, **kwargs):
        observation = self.learn_and_test(self._learner, practice_item_text,cb_mastery,cb)
        ques_H = observation[0]
        ques_text = observation[1]
        ans_H = observation[2]
        kn_emb = []
        for i in ques_H:
            ks = self.p_info[str(i)]['concepts']
            e_k = [self.conceptids[k] for k in ks]
            input_knowedge_emb = [0.] * self.num_skills
            for k in e_k:
                input_knowedge_emb[k] = 1
            kn_emb.append(input_knowedge_emb)
        msk = [1] * len(ques_H)
        state = self.KTnet.forward_state(ans_H, ques_text, kn_emb, msk)
        raw_h_cd = np.mean([i for i in cb_mastery.values()])
        h_cd = 0
        for c in cb:
            c_id = self.conceptids[c]
            h_c = state[str(c_id)]
            h_cd += h_c
        h_cd = h_cd / len(cb)

        return state, observation, h_cd, (float(state[str(target[0])]) > self.mastery_thresh_hold), ques_H, ans_H


def episode_reward(initial_score, final_score, full_score) -> (int, float):
    delta = final_score - initial_score
    normalize_factor = full_score - initial_score
    if normalize_factor == 0:
        return 0
    else:
        return delta / normalize_factor