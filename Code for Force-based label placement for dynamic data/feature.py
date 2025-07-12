class Feature:
    def __init__(self, id,position, velocity, radius=1):       
        self.id = id
        self.position = position  
        self.velocity = velocity  
        self.radius = radius  
        self.trajectory = [position]  

    def move(self, time_delta):
        self.position = (self.position[0] + self.velocity[0] * time_delta,
                         self.position[1] + self.velocity[1] * time_delta)
        self.trajectory.append(self.position)

    def get_trajectory(self):
        return self.trajectory
