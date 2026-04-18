# Robotic Manipulator Simulator

Simulation and analysis of 3-DOF robotic manipulators (RRR and PPP configurations) with trajectory tracking, workspace analysis, and link length optimization.

## Stack
- **Backend**: Python · FastAPI · NumPy · SciPy
- **Frontend**: React · Vite · Three.js · Recharts

## Project Structure
```
robotic-manipulator/
├── backend/
│   ├── kinematics/
│   │   ├── forward.py       # FK using DH parameters
│   │   ├── inverse.py       # Analytical + numerical IK
│   │   ├── trajectory.py    # Trajectory tracking + workspace check
│   │   └── optimizer.py     # Link length optimization
│   ├── main.py              # FastAPI app
│   ├── models.py            # Pydantic schemas
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── components/
    │   ├── pages/
    │   └── api/
    └── package.json
```

## Getting Started

### Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Features
- Forward kinematics via DH parameters
- Analytical inverse kinematics (elbow-up / elbow-down)
- Trajectory tracking with workspace feasibility check
- Link length optimization for unreachable trajectories
- Interactive 3D visualization
