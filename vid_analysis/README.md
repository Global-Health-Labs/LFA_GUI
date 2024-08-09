# **NAATOS LFA DATA ANALYSIS**<br>
for more details see: [Github](https://github.com/Global-Health-Labs/LFA_GUI.git)

## MicroMamba Setup
Start your VS Code and open up the Ubuntu terminal (Ctrl-Shift-~). <BR>

Get the executable using the code below: <BR>
```wget https://github.com/conda-forge/miniforge/releases/download/24.3.0-0/Mambaforge-24.3.0-0-Linux-x86_64.sh``` <BR>
Execute the file by running the code: ```bash Mambaforge-Linux-x86_64.sh```<BR>

Restart your machine to complete the setup.

## Clone Repository
Copy the link below:<BR>
```https://github.com/Global-Health-Labs/LFA_GUI/```

Open VS Code and go to the "Source Control" tab (Ctrl-Shift-G) on the left navigation bar in VS Code and click the "Clone Repository" button. 
![image3](https://github.com/Global-Health-Labs/LFA_GUI/blob/analysis_auto/readme_images/clone_repo.png)<BR>
Paste in the URL you copied and press the "Enter" key on your keyboard

## Environment Setup
Navigete to the vid_analysis repository:<BR>
```cd vid_analysis```

Run the code below to setup the environment:<BR>
```conda env create -f environment.yml```

Activate the environment using ```conda activate video_analysis```
<BR>


# Tips and Instructions
- Select the Jupyter Notebook (unit[1,2,4].ipynb OR unit[1,2,4]_w_dash.ipynb) from the Explorer tab on the left.
- Click on the "Select Kernel" icon on the top right area.
- Select "Select Another Kernel" from the pop-up menu.
- Then choose "Python Environments..." as the option.
- Select "video_analysis" from the options. 
- click the little ▶ play icon to the left of each cell OR hit ```Ctrl+Enter``` to run the code.

