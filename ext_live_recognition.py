# Divya - Extension 2: Live Video Digit Recognition
# Opens webcam and recognizes handwritten digits in real-time
# using the trained MNIST CNN model from Task 1

# import statements
import sys
import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from train_mnist import MyNetwork


# loads the trained MNIST model from file
def load_model(model_path):
    model = MyNetwork()
    model.load_state_dict(torch.load(model_path, map_location='cpu',
                                     weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


# preprocesses a frame region to match MNIST format (28x28, white on black)
def preprocess_frame(roi):
    # convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi.copy()

    # apply gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold to get clean black/white image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # resize to 28x28
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)

    # convert to tensor and normalize (same as MNIST training)
    tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
    tensor = torchvision.transforms.functional.normalize(tensor, (0.1307,), (0.3081,))

    return resized, tensor


# runs the model on a preprocessed tensor and returns prediction and confidence
def predict_digit(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.exp(output)  # convert log_softmax to probabilities
        confidence, prediction = probs.max(1)
        return prediction.item(), confidence.item(), probs.squeeze().numpy()


# draws the UI overlay on the frame
def draw_overlay(frame, prediction, confidence, all_probs, roi_box, preprocessed):
    x1, y1, x2, y2 = roi_box
    h, w = frame.shape[:2]

    # draw ROI rectangle (green box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Draw digit here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # show prediction in large text
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
    cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # show probability bar chart on the right side
    bar_x = w - 180
    bar_y_start = 30
    cv2.putText(frame, "Probabilities:", (bar_x, bar_y_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i in range(10):
        y = bar_y_start + i * 22
        bar_width = int(all_probs[i] * 120)
        bar_color = (0, 255, 0) if i == prediction else (100, 100, 100)
        cv2.rectangle(frame, (bar_x + 25, y), (bar_x + 25 + bar_width, y + 15),
                      bar_color, -1)
        cv2.putText(frame, str(i), (bar_x + 5, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{all_probs[i]:.0%}", (bar_x + 150, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # show the preprocessed 28x28 image (enlarged) in bottom-left
    preview = cv2.resize(preprocessed, (112, 112), interpolation=cv2.INTER_NEAREST)
    preview_color = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    py = h - 130
    cv2.putText(frame, "Model sees:", (10, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    frame[py:py + 112, 10:122] = preview_color

    # instructions
    cv2.putText(frame, "Press 'q' to quit", (w // 2 - 80, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return frame


# main function - opens webcam and runs live digit recognition
def main(argv):
    # load trained model
    model_path = 'mnist_model.pth'
    model = load_model(model_path)

    # open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Webcam opened successfully")
    print("Hold a handwritten digit in front of the camera")
    print("Use thick dark lines on white paper")
    print("Press 'q' to quit")

    # get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return
    h, w = frame.shape[:2]

    # define ROI (center square region)
    roi_size = min(h, w) // 2
    cx, cy = w // 2, h // 2
    x1 = cx - roi_size // 2
    y1 = cy - roi_size // 2
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # extract ROI
        roi = frame[y1:y2, x1:x2]

        # preprocess and predict
        preprocessed, tensor = preprocess_frame(roi)
        prediction, confidence, all_probs = predict_digit(model, tensor)

        # draw overlay with results
        frame = draw_overlay(frame, prediction, confidence, all_probs,
                             (x1, y1, x2, y2), preprocessed)

        # show frame
        cv2.imshow('Live Digit Recognition - MNIST CNN', frame)

        # quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done!")


if __name__ == "__main__":
    main(sys.argv)
