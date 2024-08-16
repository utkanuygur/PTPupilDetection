class TripleCombinedLoss(nn.Module):
    def __init__(self, bce_weight=1.0, circularity_weight=0.0006, connectivity_weight=1.2):
        super(TripleCombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.circularity_weight = circularity_weight
        self.connectivity_weight = connectivity_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce_loss(predictions, targets)
        circularity_loss = self.circularity_loss(predictions)
        connectivity_loss = self.connectivity_loss(predictions)

        total_loss = (
            self.bce_weight * bce_loss +
            self.circularity_weight * circularity_loss +
            self.connectivity_weight * connectivity_loss
        )
        print(f" B: {bce_loss.item()*self.bce_weight:.5f}, C: {circularity_loss.item()*self.circularity_weight:.5f}, C: {connectivity_loss.item()*self.connectivity_weight:.5f}, T: {total_loss.item()}")
        return total_loss

    def circularity_loss(self, predictions):
        probs = torch.sigmoid(predictions)
        batch_size, _, height, width = probs.size()

        contours = (probs > 0.5).float()
        cx = torch.sum(contours * torch.arange(width, device=probs.device).view(1, 1, 1, -1), dim=(2, 3)) / torch.sum(contours, dim=(2, 3))
        cy = torch.sum(contours * torch.arange(height, device=probs.device).view(1, 1, -1, 1), dim=(2, 3)) / torch.sum(contours, dim=(2, 3))

        x = torch.arange(width, device=probs.device).view(1, 1, 1, -1)
        y = torch.arange(height, device=probs.device).view(1, 1, -1, 1)
        distances = torch.sqrt((x - cx.view(batch_size, 1, 1, 1)) ** 2 + (y - cy.view(batch_size, 1, 1, 1)) ** 2)

        mean_distance = torch.sum(distances * contours, dim=(2, 3)) / torch.sum(contours, dim=(2, 3))
        std_distance = torch.sqrt(torch.sum((distances * contours - mean_distance.view(batch_size, 1, 1, 1)) ** 2, dim=(2, 3)) / torch.sum(contours, dim=(2, 3)))

        circularity_loss = torch.mean(std_distance / (mean_distance + 1e-6))
        return circularity_loss

    def connectivity_loss(self, predictions):
        probs = torch.sigmoid(predictions)
        grad_x = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        grad_y = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        connectivity_loss = torch.mean(grad_x) + torch.mean(grad_y)

        probs_binary = (probs > 0.5).float()
        num_holes = self.count_holes(probs_binary)
        hole_penalty = num_holes / (probs.size(2) * probs.size(3))

        return connectivity_loss + hole_penalty

    def count_holes(self, probs_binary):
        num_holes = 0
        for prob in probs_binary:
            num_holes += self.find_num_holes(prob)
        return num_holes

    def find_num_holes(self, prob):
        prob_np = prob.cpu().contiguous().numpy().astype(np.uint8)
        if prob_np.ndim == 3:
            num_holes = 0
            for c in range(prob_np.shape[0]):
                num_labels, labels = cv2.connectedComponents(prob_np[c], connectivity=8)
                num_holes += num_labels - 1
            return num_holes
        elif prob_np.ndim == 2:
            num_labels, labels = cv2.connectedComponents(prob_np, connectivity=8)
            return num_labels - 1
        else:
            raise ValueError("Expected a 2D or 3D tensor for connected components")
