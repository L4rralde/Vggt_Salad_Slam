from typing import List

from .sim3 import Sim3


def refine_sim3_loop(sim3_loop: List[Sim3]) -> List[Sim3]:
    sim3_seq = sim3_loop[:-1]
    constraint = sim3_loop[-1].inv()
    sim3_seq = refine_sim3_sequence(sim3_seq, constraint)
    sim3_loop[:-1] = sim3_seq
    return sim3_loop


def refine_sim3_sequence(sim3_seq: List[Sim3], constraint: Sim3) -> List[Sim3]:
    return sim3_seq
