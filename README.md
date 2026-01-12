## Updates

- 2026-01-12: Plan to add an additional upsampling branch to UNet for **Gaussian distribution modeling**, and incorporate the corresponding term into the loss function. During validation, a **false-positive suppression** formulation will be introduced.

  **Details:**
  - Add two prediction heads: a **μ (mean) head** and a **σ (variance) head**.
  - Compute **information entropy** based on the predicted Gaussian distribution.
  - The overall loss is formulated as:  
    `loss = loss + λ1 · loss_ie + λ2 · loss_sigma`.
  - During validation, apply uncertainty-aware suppression:  
    `outputs / (1 + σ)`.
