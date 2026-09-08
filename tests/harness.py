"""Inputs and HTTP helper shared by the smoke test and the benchmark."""

# Human nicotinamide N-methyltransferase, P40261.
NNMT = (
    "SGFTSKDTYLSHFNPRDYLEKYYKFGSRHSAESQILKHLLKNLFKIFCLDGVKGDLLIDIGSGPTIYQLLSA"
    "CESFKEIVVTDYSDQNLQELEKWLKKEPEAFDWSPVVTYVCDLEGNRVKGPEKEEKLRQAVKQVLKCDVTQS"
    "QPLGAVPLPPADCVLSTLCLDAACPDLPTYCRALRNLGSLLKPGGFLVIMDALKSSYYMIGEQKFSSLPLGR"
    "EAVEAAVKEAGYTIEWFEVISQSYSSTMANNEGLFSLVARKL")

# Human PD-L1 extracellular domain, Q9NZQ7.
PDL1 = (
    "AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLK"
    "DQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNA")

LIGANDS = [
    "CC(=O)Oc1ccccc1C(=O)O",                                  # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",                           # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",                             # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",                                     # paracetamol
    "OC(=O)c1ccccc1O",                                        # salicylic acid
    "CN1CCC[C@H]1c1cccnc1",                                   # nicotine
    "Clc1ccccc1C1=NCC(=O)Nc2ccc(Cl)cc12",                     # lorazepam core
    "CC(C)NCC(O)COc1ccccc1OCC=C",                             # oxprenolol
    "COc1cc2c(cc1OC)C(=O)C(CC2)Cc1ccc(OC)c(OC)c1",            # papaverine-like
    "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21",                       # chlorpromazine
    "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",            # glucose
    "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1",                          # salbutamol
]


import json
import os
import urllib.error
import urllib.request

BASE = os.environ.get("ORACLE_URL", "http://127.0.0.1:8000").rstrip("/")
TOKEN = os.environ.get("ORACLE_TOKEN", "")


def call(path, body=None, method=None, token=TOKEN, timeout=3600):
    """-> (status, decoded body)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(BASE + path, data, headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body or b"{}")
        except ValueError:      # an unhandled server error is not JSON
            return exc.code, {"detail": body.decode(errors="replace")[:500]}
