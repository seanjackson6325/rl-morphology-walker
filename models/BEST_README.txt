=== CURRENT POLICIES, FOR THE CURRENT CODE (36-value observation) ===

  warmstart_morph36.zip          fixed body. ~51 m per episode, 2.04 m/s, stride 1.07 m,
                                 knee symmetry 0.86, falls 1 episode in 30.

  LOTTERY_walk3d_morph_v2_35M.zip  random body each episode, 35M steps.
                                 On 24 unseen bodies: 17/24 falls, 428 steps, 21.33 m,
                                 7 full episodes. Well-matched legs 31.39 m,
                                 mismatched legs 11.26 m.
                                 (Copy of walk3d_morph_v2_ckpts/..._35000000_steps.zip.
                                 Chosen over the 37M final save: the run oscillated and the
                                 35M checkpoint tested best. Peak ep_len was at 34.5M.)

=== ARCHIVED, DOES NOT RUN WITH THE CURRENT CODE (35-value observation) ===

  BEST_walk3d_bothknee_v5_33M.zip  The observation grew from 35 to 36 values on 2026-08-23,
                                 because the right shin was randomised but missing from the
                                 observation. warmstart_morph36.zip is its successor, made by
                                 warmstart_morphology.py and verified equal to 2.4e-07.

=== REQUIRED CONFIG ===
  walker/mjcf.py : HIP_SEPARATION = 0.28, legs 0.55/0.55, torso 0.44 x 0.30, density 900
  walker/env3d.py: BOTHKNEE_WEIGHT = 1.6
  A policy on the wrong body looks broken. Check this first.

=== DO NOT USE ===
  walk3d_bothknee_v5.zip  late PPO collapse after the 33M checkpoint (ep_len 646 -> 96).
  Three runs in this project have lost a good policy near the end. Never trust a final save.
