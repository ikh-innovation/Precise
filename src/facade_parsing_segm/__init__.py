"""Module 1 (trained) — facade parsing via SegFormer.

Approach: instead of zero-shot prompting, this supervised path learns facade
structure directly from data. A SegFormer — a hierarchical Mix-Transformer
encoder paired with a lightweight all-MLP decoder — is fine-tuned on the CMP
Facade Database (``base/``, 606 annotated facades, 12 pixel classes). SegFormer
fits facades well: its multi-scale self-attention captures both the repeating
window/balcony grid and the broad wall regions, while staying small enough to
train and run on a single GPU. The backbone size is selectable
(``nvidia/mit-b0``…``mit-b5``) and trained with augmentation + a cosine schedule
(see ``segformer_cmp/train.py``). At inference the 12 CMP classes are reduced to
the three the pipeline consumes — facade, window (blinds merged in), door — by
``segformer_cmp.pipeline_adapter``, which mirrors the SEEM ``FacadeParser``
interface so it drops straight into ``client.py``.

Hosts :mod:`facade_parsing_segm.segformer_cmp` (training + inference scripts).
"""
