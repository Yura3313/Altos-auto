# Altos Auto

Altos Auto is an bot designed to automate gameplay in **Alto's Odyssey**. Using a neural network, it detects obstacles and attempts to navigate the game's terrain automatically.

This project is currently in development, and its performance __will__ be inconsistent.

-----

## Core Features

  * **Automated Gameplay:** Plays the game to assist with coin collection.
  * **Obstacle Detection:** Uses image processing to identify rocks and ramps.
  * **GPU accelerated:** Neural net go brrr
  * **Automated Actions:** Executes jumps and backflips in response to obstacles.

-----

## Getting Started

### **Step 1: Prerequisites**

Make sure you have **Python 3.13** (or newer) and pip installed on your system. It will probably work on older versions, but i only tested on 3.13.7

### **Step 2: Installation**

Clone the repository and install the required packages using the `requirements.txt` file. Please note that the `torch` packages are large and may take some time to download.

```bash
pip install -r requirements.txt
```

### **Step 3: Game Configuration**

The bot requires the game to be opened fullscreen on the main monitor.

### **Step 4: Running the Bot**

Open a terminal, navigate to the project directory, and execute the main script.

```bash
python main.py
```

To stop the bot, press `Esc` while focused on the preview window.

-----

## How It Works

The bot operates by repeatedly capturing screenshots of the game window. It runs the screenshots through a neural network. Based on the output, it automagically triggers keypresses to make the character jump.

-----

## Configuration and Contributing

### **Known Issues**

  * Object detection is not always precise, which may cause missed jumps or random jumps.
  * The script is highly sensitive to the game window's size and placement.
  * ...

### **Customization**

Key parameters, such as jump sensitivity and the screen region for detection, can be adjusted within the source code. Modifying these values may improve performance.

### **Contributing**

This is an early-stage project, and contributions are welcome. If you have ideas for improvements, open an issue. If you would like to fix a bug, feel free to submit a pull request - though the code is very messy, so good luck. If you have nothing to do you can record your gameplay using the gameplay recorder and train a new model :).
