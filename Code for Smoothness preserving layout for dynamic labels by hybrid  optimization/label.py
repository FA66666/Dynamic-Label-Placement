class Label:
    def __init__(self, id, feature, position, length=40,width=16, velocity=(0, 0), priority=0):
        self.id = id
        self.feature = feature
        self.position = position
        self.length = length
        self.width = width
        self.velocity = velocity
        self.priority = priority
        self.trajectory = [position]

    def move(self, time_delta, forces):
        total_force = [0, 0]
        for force in forces.values():
            total_force[0] += force[0]
            total_force[1] += force[1]
        acceleration = (total_force[0], total_force[1])
        self.velocity = (self.velocity[0] + acceleration[0] * time_delta,
                         self.velocity[1] + acceleration[1] * time_delta)
        self.position = (self.position[0] + self.velocity[0] * time_delta,
                         self.position[1] + self.velocity[1] * time_delta)
        self.trajectory.append(self.position)

    def get_trajectory(self):
        return self.trajectory


