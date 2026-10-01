# PADS `device_id` review and correction

Date: 2026-09-16

## Conclusion

The PADS publication does **not** describe a mixed-device acquisition. Its Methods
section states that every participant wore two Apple Watch Series 4 devices. The
PhysioNet dataset description repeats the same statement.

However, the released PhysioNet v1.0.0 observation JSON files contain two distinct
`device_id` strings: `Apple Watch Series 3` and `Apple Watch Series 4`. This is an
official-source inconsistency. Without clarification from the dataset authors, the
JSON field must not be equated with confirmed physical hardware generation.

## Verified evidence

1. Peer-reviewed article:
   <https://doi.org/10.1038/s41531-023-00625-7>
   states: all participants wore two Apple Watch Series 4 smartwatches.
2. PhysioNet v1.0.0 documentation:
   <https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/>
   also describes two Apple Watch Series 4 smartwatches.
3. Official PhysioNet files:
   - `movement/observation_004.json` contains `Apple Watch Series 3`.
   - `movement/observation_235.json` contains `Apple Watch Series 4`.
4. The server copies match the official PhysioNet checksums:
   - observation 004: `bd33881198778bd73e0179af01169d0aae8cf1ac061a740451286ba898f2e7cf`
   - observation 235: `0f340226b54942125215836d165bc77b2e0e99ccfc41617fb84dc65d312ea707`
5. The published author correction
   <https://doi.org/10.1038/s41531-024-00710-5>
   addresses Fig. 3 and questionnaire text, not device metadata.

## Revised interpretation for this project

- Withdrawn: “PADS has confirmed Series-3-versus-Series-4 hardware confounding.”
- Retained: the released `device_id` metadata is associated with diagnosis and is a
  useful negative-control/acquisition-batch proxy.
- Not justified: calling `device_id`-stratified results cross-device generalization,
  or using this field as a confirmed hardware-domain target.
- Required before a hardware claim: clarification from the PADS authors or original
  acquisition logs that establish what the two metadata values mean.

The previously computed metadata-only BA/AUROC of about 0.6203 is preserved as an
audit result, but is relabelled `device_id`-only negative control. No model is selected
or rejected on the basis of that value alone, and device-adversarial training is not
started from this unverified metadata field.
