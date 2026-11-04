"""Paired episodes with disjoint note pools and no replacement."""
import random
from .span_dataset import FewShotEpisode

class HeldoutEpisodeBuilder:
    def __init__(self, support, query, entity_types, n_query=50, seed=42):
        self.support, self.query = support, query
        self.entity_types, self.n_query, self.seed = list(entity_types), n_query, seed
        if {x.note_id for x in support.all_items()} & {x.note_id for x in query.all_items()}:
            raise ValueError('Support/query note leakage')

    def build_episodes(self, n_shots, n_episodes=100):
        episodes = []
        for i in range(n_episodes):
            support, query = {}, []
            for j, label in enumerate(self.entity_types):
                sp, qp = list(self.support.get_by_type(label)), list(self.query.get_by_type(label))
                if len(sp) < n_shots or len(qp) < self.n_query:
                    raise ValueError(f'Insufficient {label} spans: support={len(sp)}, query={len(qp)}')
                # Separate generators keep query and nested support identical across K.
                random.Random(self.seed + i*1009 + j).shuffle(sp)
                random.Random(self.seed + i*1009 + j + 1000000).shuffle(qp)
                support[label] = sp[:n_shots]; query.extend(qp[:self.n_query])
            episodes.append(FewShotEpisode(support,query,self.entity_types,n_shots,i))
        return episodes
