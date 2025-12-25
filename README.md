<p align="center">
<picture>
  <img width="300" height="300" alt="icon" src="https://github.com/user-attachments/assets/76bc815f-fbc0-4edc-af32-8183eb03aba0" />
  <h1 align='center'>Camera Control</h1>
  <p align='center'>Beautiful GUI to control your camera remotely.</p>
</picture>
</p>

## Installation
Before your get started, check if your camera is supported by gphoto2: http://www.gphoto.org/proj/libgphoto2/support.php
1. Install gphoto2
   Linux:
    ```bash
    sudo apt install gphoto2
    sudo apt install libgphoto2-6
    ```
    macOS:  
    ```bash
    brew install gphoto2
    ```
2. Clone this repository:
   ```bash
    git clone https://github.com/valentinfrlch/camera-control
    ```
3. Install Python dependencies (tested with python 3.11)
    ```bash
    cd camera-control
    pip install -r requirements.txt
    ```
4. Connect your camera via USB
5. Run webserver
    ```bash
    python webserver.py
    ```
6. You should see the following in your terminal:
   ```bash
   * Serving Flask app 'webserver'
   * Debug mode: on
   WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
   * Running on all addresses (0.0.0.0)
   * Running on http://127.0.0.1:8000
   * Running on http://10.0.1.220:8000
   ```
   The web interface should automatically open in your web browser. To control your camera from your phone, make sure your phone and computer are on the same network, then open the second url displayed in the terminal (in this case `http://10.0.1.220:8000`) in your phone's browser.

   
### Troubleshooting
If you have trouble connecting your camera, close any applications that may be using it. On macOS, open Activity Monitor, search for ptpcamera, and quit any related processes—these will prevent gphoto2 from accessing the camera.
