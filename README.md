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
  - **2026-01-14:** Feature maps **P2, P3, and P4** are spatially aligned to the original image resolution and jointly supervised using **Focal Loss** during training. During validation, only **P3** is used for prediction.

  **Results:**
  - **2026-01-14:** It was confirmed that the backbone outputs feature levels **P2, P3, P4, P5, and P6** by default.
  - The feature map **key names have been aligned consistently** within the implementation.
  - The complete model pipeline now **runs successfully on the local machine**, indicating that the overall framework has been correctly integrated.

  - **ResNet Bottleneck:**
    - Conv1(x) = ReLU(BN(Conv1x1(x)))
    - Conv2(x) = ReLU(BN(Conv3x3(x)))
    - Conv3(x) = BN(Conv1x1(x))
    - Bottleneck(x) = ReLu(Conv3(Conv2(Conv1(x))) + S(x))
    
  - **Image preprocessing:** 
    - f(x) = Tensor(Transform(img)) = y
    - g(y) = MaxPool(ReLU(BN(Conv7x7(y)))) = z
    
  - **ResNet-50:**
    - C2 = 3 × Bottleneck(z)
    - C3 = 4 × Bottleneck(z)
    - C4 = 6 × Bottleneck(z)
    - C5 = 3 × Bottleneck(z)

    - P5 = Conv1x1(C5)
    - P4 = Conv1x1(C4) + Up(C5)
    - P3 = Conv1x1(C3) + Up(C4)
    - P2 = Conv1x1(C2) + Up(C3)
