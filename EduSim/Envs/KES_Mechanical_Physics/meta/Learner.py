
import os
import copy
import random

import numpy as np
from EduSim.Envs.meta import MetaLearner, MetaInfinityLearnerGroup
import json
import os.path as osp


class Learner(MetaLearner):
    def __init__(self,
                 initial_log,
                 learning_target: set,
                 _id=None,
                 seed=None):
        super(Learner, self).__init__(user_id=_id)

        self._target = learning_target
        self._logs = initial_log
        self._state = []
        self.random_state = np.random.RandomState(seed)

    def update_logs(self, logs):
        self._logs = logs

    @property
    def profile(self):
        return {
            "id": self.id,
            "logs": self._logs,
            "target": self.target
        }

    def learn(self, learning_item, score):
        
        self._logs[0].append(0)
        self._logs[1].append(learning_item)
        self._logs[2].append(score)

    @property
    def state(self):
        return self._state

    def response(self, test_item) -> ...:
        return self._state[test_item]

    @property
    def target(self):
        return self._target




class KESPhysicsLeanerGroup(MetaLearner):
    def __init__(self, dataRec_path, seed=2024):
        super(KESPhysicsLeanerGroup,self).__init__()
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)
        problem_info_path = self.data_path.replace("data_rec.txt", "problem_info.json")
        if not os.path.exists(problem_info_path):
            problem_info_path = self.data_path.replace("records.txt", "problem_info.json")
        if not os.path.exists(problem_info_path):
            import os.path as osp
            base_dir = osp.dirname(self.data_path)
            problem_info_path = osp.join(base_dir, "problem_info.json")
        try:
            with open(problem_info_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content.startswith('{') and content.count('{') > 1:
                    brace_count = 0
                    end_pos = 0
                    for i, char in enumerate(content):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    content = content[:end_pos]
                self.p_info = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse problem_info.json at {problem_info_path}: {e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"problem_info.json not found. Tried: {problem_info_path}")

    def __next__(self):
        usr = []
        length = []
        qs = []
        Ans = []
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 3):
                usr.append((i+1)%3)
                length.append(int(lines[i].strip()))
                qs.append(list(map(int, lines[i + 1].strip().split())))
                Ans.append(list(map(int, lines[i + 2].strip().split())))

        qs_texts = []
        for q in qs:
            qs_text = []
            for i in q:
                qs_text.append(self.p_info[str(i)]["content"])
            qs_texts.append(qs_text)

        all_students = len(usr)
        index = self.random_state.randint(all_students)
        stu_qdata = qs[index]
        stu_qtdata = qs_texts[index]
        stu_ansdata = Ans[index]
        session = [stu_qdata[:int(0.6*len(stu_qdata))], stu_qtdata[:int(0.6*len(stu_qdata))], stu_ansdata[:int(0.6*len(stu_qdata))]]
        learning_targets = list(self.random_state.choice(stu_qdata[int(0.8*len(stu_qdata)):],1))

        initial_log = copy.deepcopy(session)

        return Learner(
            initial_log=initial_log,
            learning_target=learning_targets,
        )
