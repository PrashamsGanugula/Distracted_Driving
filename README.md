# Real-Time Driver Distraction Monitor

## 1. Project Goal

This project implements a deep learning pipeline to detect driver distraction from video feeds. The system classifies the driver's state into one of four categories in real-time:

* **Normal Driving**
* **Looking Left**
* **Looking Right**
* **Using a Phone**

The entire workflow, from data preparation to model training and testing, is managed through a series of Jupyter Notebooks. A key feature of this project is the use of an AI-powered automated labeling system to generate the initial training dataset from raw, uncut videos.

---

## 2. Tech Stack

* **Language:** Python 3.10
* **Environment:** Anaconda
* **Core Libraries:**
    * **TensorFlow / Keras:** For building and training the neural network.
    * **OpenCV:** For video processing and frame extraction.
    * **Hugging Face Transformers:** For using the pre-trained `openai/clip-vit-large-patch14` model for automated data labeling.
    * **MoviePy:** For cutting and creating video clips.
    * **Scikit-learn:** For splitting the dataset and performance metrics.
    * **Pandas, NumPy, Matplotlib, Seaborn:** For data handling and visualization.

---

## 3. Project Structure

The project is organized into three distinct Jupyter Notebooks for a clear and modular workflow.

```
└── DriverDistractionProject/
    ├── raw_videos/                 # Folder for your raw, uncut videos
    │   ├── driver_01.mp4
    │   └── ...
    ├── Model_Results/              # Auto-generated to store the trained model and graphs
    │   ├── best_model.keras
    │   └── training_plots.png
    ├── Normal_Driving/             # Auto-generated folder with labeled clips
    ├── Looking_Left/               # Auto-generated folder with labeled clips
    ├── Looking_Right/              # Auto-generated folder with labeled clips
    ├── Using_Phone/                # Auto-generated folder with labeled clips
    ├── 1-Automated-Labeler.ipynb   # Notebook 1: Creates the dataset
    ├── 2-Model-Training.ipynb      # Notebook 2: Trains the model
    └── 3-Live-Prediction.ipynb     # Notebook 3: Tests the model
```

---

## 4. Setup and Installation

Follow these steps to set up the project on your local machine.

**Prerequisites:** You must have **Anaconda** installed.

1.  **Create the Conda Environment:**
    Open the Anaconda Prompt and create a new environment for the project.
    ```bash
    conda create -n driver_project_final python=3.10
    ```

2.  **Activate the Environment:**
    ```bash
    conda activate driver_project_final
    ```

3.  **Install All Required Libraries:**
    With the environment active, run this single command to install everything.
    ```bash
    pip install jupyter notebook tensorflow "numpy<2.2.0" pandas scikit-learn seaborn opencv-python "moviepy==1.0.3" transformers torch accelerate tf-keras
    ```

---

## 5. How to Run the Project

The project must be run in the following sequence:

#### **Step 1: Generate the Dataset**

1.  Place your raw video files inside the `raw_videos` folder.
2.  Open the `1-Automated-Labeler.ipynb` notebook.
3.  **Crucially, update the `BASE_DIR` variable** to the absolute path of your `DriverDistractionProject` folder.
4.  Run all cells in the notebook. This will take a long time as it processes all videos, classifies actions, and saves the resulting clips into new labeled folders.

#### **Step 2: Train the Model**

1.  Open the `2-Model-Training.ipynb` notebook.
2.  Update the `DATASET_DIR` variable to your project folder's path.
3.  Run all cells. This will load the clips, train the `ConvLSTM` model, and save the best version as `best_model.keras` in the `Model_Results` folder. It will also display the final accuracy and performance graphs.

#### **Step 3: Test the Model**

1.  Place a new, short test video (5-10 seconds) inside the main `DriverDistractionProject` folder.
2.  Open the `3-Live-Prediction.ipynb` notebook.
3.  Update the `BASE_DIR` and `test_video_path` variables.
4.  Run all cells. The notebook will display the test video with the model's prediction and confidence score.

---

## 6. Model Architecture

The project uses a **Convolutional Long Short-Term Memory (ConvLSTM)** network. This architecture is ideal for video classification because it can learn both spatial features (what's in an image) and temporal features (how those features change over time).

* **Convolutional Layers:** Extract visual features from each frame.
* **LSTM Layers:** Analyze the sequence of features from the frames to understand the action.
* **Dense Layers:** Classify the sequence into one of the final categories.

This allows the model to differentiate between subtle movements and sustained actions, which is key for distraction detection.
