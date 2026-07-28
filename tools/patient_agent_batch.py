import sys
sys.path.append("/data1/huangguolin/workplace2/graph_agent/graph_agent_v5/")
import json
import os
from util.utils import *
from tools.evopatient import get_evopatient_simulator


with open('prompt/patient_agent/response_batch.txt', 'r', encoding='utf-8') as file:
    query_ori = file.read()

def _evopatient_generator(prompt, max_tokens, temperature):
    return patient_agent(
        max_token=max_tokens,
        temperature=temperature,
        system_role="",
        user_input=prompt,
    )


def patient_response(
    history,
    single_patient_condiction,
    question,
    patient_key=None,
    stage=None,
):
    patient_backend = os.getenv("PATIENT_BACKEND", "native").strip().lower()
    if patient_backend == "evopatient":
        simulator = get_evopatient_simulator()
        return simulator.answer(
            case_id=patient_key,
            question=question,
            initial_presentation=single_patient_condiction,
            fallback_record=history,
            stage=stage,
            generator=_evopatient_generator,
        )
    if patient_backend != "native":
        raise ValueError(
            f"Unsupported PATIENT_BACKEND={patient_backend!r}; "
            "expected native or evopatient."
        )
    query_prompt = query_ori.replace("{history}", history).replace("{question}", question).replace("{patient_condiction}", single_patient_condiction)
    # print(query_prompt)
    response = patient_agent(max_token=10000, temperature=0, system_role='', user_input=query_prompt)
    return response








