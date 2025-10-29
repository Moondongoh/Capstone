from ultralytics import YOLO

model = YOLO(r"C:\GIT\Capstone\Web\flask-template\weights\best.pt")
model.export(format="onnx")
print("Model has been successfully converted to ONNX format.")
