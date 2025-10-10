## **Project OverView**

This project demonstrates an end-to-end **Edge AI computer vision** application built on Linux and to be deployed on a **Raspberry Pi 5 using Docker**. 
The container runs a **GUI-based application** (not web-based) built with **PySide6**, providing a local user interface for live camera preview and on-device inference. 
The system uses a **GStreamer video pipeline** integrated with **ONNX Runtime** to perform real-time object detection using open-source **YOLO models**, 
with detection overlays rendered directly on the GUI window.. 
The workflow supports, containerized deployment, and support for model updates through mounted **ONNX artifacts** — illustrating a complete path from 
prototype to production-ready application.

### **Project Delivery Pipeline**

The application’s development lifecycle extends through containerization and publishing of the final image to **GitHub Container Registry (GHCR)**, 
enabling reproducible and portable deployment across devices. Actual delivery and deployment to the **Raspberry Pi 5** are handled through a companion repository — 
**Pi5_App_Model_CD_pipeline** https://github.com/tonu1990/Pi5_App_Model_CD_pipeline  — which provides a reusable **GitHub Actions template** for edge application delivery. 
This workflow leverages a **self-hosted runner** on the Pi 5 to execute CI/CD jobs locally, pulling the latest **multi-architecture (amd64/arm64) Docker image from GHCR**, 
generating a lightweight **launcher script**, and installing **click-to-run desktop and menu icons** for direct GUI execution.
