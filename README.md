## Updates

- 2026-01-12: Plan to add an additional upsampling branch to UNet for **Gaussian distribution modeling**, and incorporate the corresponding term into the loss function. During validation, a **false-positive suppression** formulation will be introduced.

  **Details:**
  - Add two prediction heads: a **μ (mean) head** and a **σ (variance) head**.
  - Compute **information entropy** based on the predicted Gaussian distribution.
  - The overall loss is formulated as:  
    `loss = loss + λ1 · loss_ie + λ2 · loss_sigma`.
  - During validation, apply uncertainty-aware suppression:  
    `outputs / (1 + σ)`.
  
  **Results:**
  - The fixed weighting coefficients (λ) were replaced by the **ratio between each auxiliary loss and the prediction loss**.
  - Preliminary results were obtained around **2026-01-13**. The achieved **F1-score is 0.4201**, which is **inferior to the baseline UNet**.

- 2026-01-13: Plan to study and integrate **RetinaNet**, with the goal of understanding its design principles and exploring potential structural modifications for performance improvement.

  **Details:**
  - A **ResNet-50 + FPN** architecture is adopted as the backbone.
  - The **RetinaNet head** is directly connected to the aligned FPN feature maps at levels **P3–P5**.
  - Training is performed using **Focal Loss** for classification.
  - The current framework remains preliminary and requires further refinement. A planned extension is to **incorporate the P2 feature level** to better capture small-scale targets.
