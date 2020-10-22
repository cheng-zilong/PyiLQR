#%%
import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
p.resetSimulation()
p.setGravity(0, 0, -1e2)
targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
steeringSlider = p.addUserDebugParameter("steering", -0.5, 0.5, 0)
p.setPhysicsEngineParameter(enableConeFriction=0) 
floor = p.loadURDF("plane.urdf")
car = p.loadURDF("pybullet\\racecar.urdf", globalScaling=15,basePosition = [0,0,1])
#%%
useRealTimeSim = 0

#for video recording (works best on Mac and Linux, not well on Windows)
#p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim)  # either this
#p.loadURDF("plane.urdf")

for i in range(p.getNumJoints(car)):
  print(p.getJointInfo(car, i))
inactive_wheels = [3, 5]
wheels = [0, 1]

for wheel in inactive_wheels:
  p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

steering = [2, 4]


while (True):
  targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
  steeringAngle = p.readUserDebugParameter(steeringSlider)
  #print(targetVelocity)
  p.getContactPoints()
  for wheel in wheels:
    p.setJointMotorControl2(car,
                            wheel,
                            p.VELOCITY_CONTROL,
                            targetVelocity=targetVelocity,
                            force=1e10)

  for steer in steering:
    p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=steeringAngle)

  if (useRealTimeSim == 0):
    p.stepSimulation()
  time.sleep(0.01)
# %%
