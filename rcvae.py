# Compatibility shim for legacy pretrained augmentation checkpoints.
#
# The pickle files in pretrain_model/ (icvae_Ponzi.pkl, icvae_Phish.pkl) were
# saved with torch.save(<full object>) when the augmentation module was named
# `rcvae` and its top class was `RCVAE`. The codebase later renamed this to
# icvae.py / ICVAE (same architecture: Encoder + Decoder). Unpickling a full
# object requires the ORIGINAL module and class names to be importable, so this
# module re-exposes them, mapping onto the current implementation. This lets
# torch.load restore the existing pretrained weights unchanged.
from icvae import Encoder, Decoder, ICVAE as RCVAE  # noqa: F401
