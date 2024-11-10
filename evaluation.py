from sklearn.metrics import precision_score, recall_score
import torch

def evaluate_model(model, test_loader):
    """
    Evaluate model performance on a test dataset.
    - Calculates precision and recall.

    Args:
        model (torch.nn.Module): Trained model for evaluation.
        test_loader (DataLoader): DataLoader object for test dataset.

    Returns:
        None
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []  # To store all true labels
    all_preds = []   # To store all predictions

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  # Get the predicted class
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate precision and recall scores
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}')
