# Recommendation: utility-facing summary and Section 4 answers

## 4. Practical implications

### Opportunity

Offer an opt-in, household-level washing-machine usage summary that helps customers understand how often the appliance runs and the approximate load associated with detected operating episodes. If a customer is interested, this could support a practical conversation about avoiding unnecessary partial loads or using lower-energy settings. The current model is not accurate enough for individual appliance estimates or billing, so this initial insight should be based on the directly monitored REFIT appliance readings, not on model predictions.

### How to measure impact

Pilot the feedback with consenting customers and a comparable control group. Measure change in whole-home kWh per household-week and, where appliance-level monitoring is available, washing-machine kWh per operating episode and episode frequency. Compare a pre-intervention baseline with a follow-up period, adjust for season and occupancy where possible, and track engagement. Distinguish genuine energy reduction from simply shifting laundry to another time; do not claim savings from a change in timing alone.

### Limits and transferability

REFIT has only 20 UK homes and household/appliance usage varies. Whole-home aggregate readings contain overlapping appliance signatures, so a model can confuse simultaneous loads. Missing data, long outages, official forward-filling, removal of IAM spikes above 4,000 W, and the official `Issues` discrepancies affect both labels and evaluation. The evaluation held out one home, and its Random Forest did not consistently beat the always-off baseline; the result is not ready for deployment.

UK home size, appliances, climate, occupancy, tariffs, and daily routines may differ from India. A model trained on REFIT should not be assumed to transfer; it would need local smart-meter data, local appliance labels, and evaluation on homes held out from training in the target region.

### If only 15- or 30-minute smart-meter data were available

Longer intervals would average away short washing-machine power changes and combine the machine with other household loads. The timing and duration of operating episodes would become less precise, and multiple appliance events could land in the same interval. Cycle detection and disaggregation would therefore be substantially less reliable than at one-minute resolution; the model would need to be retrained and evaluated at the actual meter interval.
