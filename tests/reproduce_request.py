from __future__ import annotations
import argparse, json
from urllib.request import Request, urlopen

def post(base_url: str, payload: dict) -> dict:
    data=json.dumps(payload).encode()
    req=Request(base_url.rstrip("/")+"/v1/systemone",data=data,headers={"Content-Type":"application/json"})
    with urlopen(req,timeout=120) as response:return json.load(response)

def validate(response: dict) -> None:
    answers=response["answers"]
    choice=answers["route"]
    assert choice["type"]=="choice"
    assert set(choice["probabilities"])=={"账单","shipping","tech-support"}
    assert choice["choice"] in choice["probabilities"]
    probs=choice["probabilities"]
    assert all(isinstance(v,(int,float)) and 0<=v<=1 for v in probs.values())
    assert abs(sum(probs.values())-1)<0.01
    assert probs[choice["choice"]]==max(probs.values())
    noul=answers["urgent"]
    assert noul["type"]=="noul" and 0<=noul["noul"]<=1

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--base-url",default="http://127.0.0.1:18080");a=p.parse_args()
    payload={"model":"IF-decision-27b-v1","state":{"ticket":"Charged twice and order remains unpaid."},"questions":{
      "route":{"type":"choice","instructions":"Choose the handling team.","criteria":{"账单":"Billing and refunds","shipping":"Delivery","tech-support":"Application support"}},
      "urgent":{"type":"noul","instructions":"Should this be handled today?"}}}
    out=post(a.base_url,payload);validate(out)
    print(json.dumps(out,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":main()
