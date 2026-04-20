# 🌙 DriftDreamer

## 🚀 What is this project?

DriftDreamer is a project where a robot learns to move on its own inside a simulated warehouse.

Instead of hardcoding rules, the robot learns a **world model** — basically a mini simulator in its head. It then uses this model to plan its actions, follow a path, and avoid obstacles.

The project is inspired by ideas from modern AI systems like Dreamer and self-driving research.

> This project is still under development, and I am continuing to improve the model, planning, and overall performance.

## 🧠 What does it do?

- Drives a tugbot around a warehouse in Gazebo
- Learns how its own movements change its position
- Uses lidar (distance sensors) to detect nearby obstacles
- Uses a reward function to decide what good behavior looks like
- Plans step by step to follow a route, avoid crashing, and reach the goal

## ⚙️ How it works

The project has three main steps.

### 1. Collect Data

The robot drives around the simulator and records:

- its position
- its actions
- what happens next
- lidar readings

This becomes the training data.

### 2. Learn a World Model

A neural network is trained to answer one question:

> "If I take this action, what will happen next?"

After training, the robot has a learned simulator in its head.

### 3. Plan Actions with Rewards

At every step, the robot:

1. Looks at its current state
2. Tries many possible action sequences inside its world model
3. Scores them using a **reward function**
4. Picks the action sequence with the best score
5. Executes the first action
6. Repeats

The reward helps the robot prefer actions that move it toward the route, keep it aligned, and avoid obstacles. 

## ▶️ Demo

<!--
HOW TO EMBED Demo.mp4 (it's too big for git but GitHub can host it for free):
  1. After pushing this repo, open any Issue on github.com/Rakshit-p/DriftDreamer
     (or an empty PR comment — you don't have to submit it).
  2. Drag and drop Demo.mp4 into the comment box. Wait for the upload.
  3. GitHub replaces it with a URL like
        https://github.com/user-attachments/assets/<long-uuid>
  4. Copy that URL and replace the line below with it, then commit + push.
  5. GitHub's markdown auto-renders bare user-attachments video URLs
     as an inline video player — no <video> tag needed.
-->

<!-- replace the next line with the user-attachments URL from step 3 -->


## 📦 Key Features

- Learned world model
- Uses lidar for obstacle awareness
- Reward-guided planning
- Model Predictive Control (MPC)
- Runs fully on a laptop
- Inspired by real-world self-driving research

## 🛠️ Current Status

This project is still a work in progress. I am actively improving the training setup, planning quality, and obstacle handling.
