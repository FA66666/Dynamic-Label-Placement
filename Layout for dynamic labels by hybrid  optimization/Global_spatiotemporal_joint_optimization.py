import math
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches

paramsA1 = {
    'Wlabel-label': 80,
    'Wlabel-feature': 50,
    'Worient': [1, 2, 3, 4],  
    'Wdistance': 20,
    'Wout-of-axes': 320,
    'Wintersect': 1,  
    'Wradius': 20,
    'Wangle': 10,
    'delta_t': 1  
}

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

class LabelOptimizer:  
    def __init__(self, labels, features, params, max_x=1000, max_y=1000):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = {}
        self.joint_sets = []
        self.max_x = max_x  
        self.max_y = max_y  

    def calculate_label_label_overlap(self, i, j, label_positions):  
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]
        l1, w1 = self.labels[i].length, self.labels[i].width
        l2, w2 = self.labels[j].length, self.labels[j].width

        rect1 = {
            'x_min': x1,
            'x_max': x1 + w1,
            'y_min': y1,
            'y_max': y1 + l1
        }
        rect2 = {
            'x_min': x2,
            'x_max': x2 + w2,
            'y_min': y2,
            'y_max': y2 + l2
        }

        overlap_x = max(0, min(rect1['x_max'], rect2['x_max']) - max(rect1['x_min'], rect2['x_min']))
        overlap_y = max(0, min(rect1['y_max'], rect2['y_max']) - max(rect1['y_min'], rect2['y_min']))
        return overlap_x * overlap_y

    def calculate_label_feature_overlap(self, label_idx, feature_idx, label_positions):
        label_pos = label_positions[label_idx]
        feature_pos = self.features[feature_idx].position
        feature_radius = self.features[feature_idx].radius
        label_width = self.labels[label_idx].width
        label_length = self.labels[label_idx].length

        rect_corners = [
            (label_pos[0] - label_length/2, label_pos[1]-  label_width/2),
            (label_pos[0] + label_length/2, label_pos[1]-  label_width/2),
            (label_pos[0] - label_length/2, label_pos[1] + label_width/2),
            (label_pos[0] + label_length/2, label_pos[1] + label_width/2)
        ]

        min_dist = float('inf')
        for i in range(4):
            p1 = rect_corners[i]
            p2 = rect_corners[(i+1)%4]
            dist = self.point_to_line_distance(feature_pos, p1, p2)
            min_dist = min(min_dist, dist)

        if min_dist > feature_radius:
            return 0

        
        overlap_area = self.numerical_integration_overlap(
            feature_pos, feature_radius,
            label_pos, label_width, label_length
        )

        return overlap_area

    def point_to_line_distance(self, point, line_start, line_end):
       
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        l2 = (x2-x1)**2 + (y2-y1)**2

        if l2 == 0:
            return math.hypot(x0-x1, y0-y1)

        t = max(0, min(1, ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1))/l2))

        projection_x = x1 + t*(x2-x1)
        projection_y = y1 + t*(y2-y1)

        return math.hypot(x0-projection_x, y0-projection_y)

    def numerical_integration_overlap(self, circle_center, circle_radius, rect_pos, rect_width, rect_height):
       
        dx = 0.01
        dy = 0.01

        overlap_area = 0
        x_start = max(rect_pos[0], circle_center[0] - circle_radius)
        x_end = min(rect_pos[0] + rect_width, circle_center[0] + circle_radius)
        y_start = max(rect_pos[1], circle_center[1] - circle_radius)
        y_end = min(rect_pos[1] + rect_height, circle_center[1] + circle_radius)

        for x in np.arange(x_start, x_end, dx):
            for y in np.arange(y_start, y_end, dy):
                in_circle = (x - circle_center[0])**2 + (y - circle_center[1])**2 <= circle_radius**2
                in_rect = (x >= rect_pos[0] - rect_height/2 and x <= rect_pos[0] + rect_height/2 and
                          y >= rect_pos[1] -  rect_width/2 and y <= rect_pos[1] + rect_width/2)

                if in_circle and in_rect:
                    overlap_area += dx * dy

        return overlap_area

    def get_quadrant(self, theta):
       
        
        
        theta = theta % (2 * math.pi)
        
        
        if 0 <= theta < math.pi/2:
            return 1  
        elif math.pi/2 <= theta < math.pi:
            return 2  
        elif math.pi <= theta < 3*math.pi/2:
            return 3  
        else:
            return 4  

    def cartesian_to_polar(self, cartesian):
       
        x, y = cartesian
        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        return r, theta

    def lines_intersect(self, line1, line2):
       

        def ccw(A, B, C):
           
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = line1
        C, D = line2

        
        if not all(isinstance(p, tuple) and len(p) == 2 for p in [A, B, C, D]):
            raise ValueError("每个点必须是 (x, y) 元组")

        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def calculate_leader_intersections(self, label_positions):
       
        intersections = 0
        valid_positions = [pos for pos in label_positions if pos is not None]

        for i in valid_positions:
            for j in valid_positions:
                line1 = (label_positions[i], self.features[i].position)
                line2 = (label_positions[j], self.features[j].position)

                if self.lines_intersect(line1, line2):
                    intersections += 1

        return intersections

    def check_out_of_axes(self, label_positions):
       
        total_area = 0
        for i, pos in enumerate(label_positions):
            x, y = pos
            label = self.labels[i]
            label_width = label.width
            label_height = label.length

            clipped_left = max(0, -x)
            clipped_right = max(0, (x + label_width) - self.max_x)
            clipped_top = max(0, -y)
            clipped_bottom = max(0, (y + label_height) - self.max_y)

            total_area += (
                clipped_left * label_height +
                clipped_right * label_height +
                clipped_top * label_width +
                clipped_bottom * label_width
            )

        return total_area

    def check_trajectory_intersection(self, feature_i, feature_j):
       
        traj_i = self.features[feature_i].trajectory
        traj_j = self.features[feature_j].trajectory

        for t in range(len(traj_i) - 1):
            line1 = (traj_i[t], traj_i[t + 1])
            line2 = (traj_j[t], traj_j[t + 1])
            if self.lines_intersect(line1, line2):
                return True
        return False

    def detect_joint_sets(self):
       
        joint_sets = []
        max_frames = len(self.features[0].trajectory) if self.features else 0

        for t in range(max_frames):
            current_set = set()

            
            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    has_interaction = False
                    for dt in range(min(self.params['delta_t'], max_frames - t)):
                        future_frame = t + dt
                        pos_i = self.features[i].trajectory[future_frame]
                        pos_j = self.features[j].trajectory[future_frame]

                        dx = pos_i[0] - pos_j[0]
                        dy = pos_i[1] - pos_j[1]
                        distance = math.hypot(dx, dy)

                        if distance < 2: 
                            has_interaction = True
                            break

                    if has_interaction:
                        current_set.update({i, j})

            if current_set:
                complexity = self.calculate_joint_set_complexity(current_set, t)
                
                
                feature_positions = {}
                for feat_idx in current_set:
                    feature_positions[feat_idx] = self.features[feat_idx].trajectory[t]
                
                
                print(f"\n--- Joint Set at Frame {t} (Complexity: {complexity:.2f}) ---")
                feature_list = list(current_set)
                for idx1 in range(len(feature_list)):
                    for idx2 in range(idx1 + 1, len(feature_list)):
                        feat1_idx = feature_list[idx1]
                        feat2_idx = feature_list[idx2]
                        pos1 = feature_positions[feat1_idx]
                        pos2 = feature_positions[feat2_idx]
                        dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                        print(f"  Distance between F{feat1_idx} and F{feat2_idx}: {dist:.2f}")
                
                
                joint_sets.append({
                    'set': current_set,
                    'frame': t,
                    'complexity': complexity,
                    'feature_positions': feature_positions  
                })

        
        self.joint_sets = sorted(joint_sets, key=lambda x: x['complexity'], reverse=True)

    def calculate_joint_set_complexity(self, feature_set, frame):
       
        complexity = 0
        features_list = list(feature_set)

        
        positions = [self.features[i].trajectory[frame] for i in features_list]
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        density = len(feature_set) / area if area > 0 else float('inf')

        
        intersections = 0
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                if self.check_trajectory_intersection(features_list[i], features_list[j]):
                    intersections += 1

        
        complexity = density * (1 + intersections)
        return complexity

    def calculate_static_energy(self, label_positions, joint_set):
       
        E_overlap = 0
        E_position = 0
        E_aesthetics = 0
        E_constraint = 0

        current_features = list(joint_set['set'])
        
        feature_positions = joint_set.get('feature_positions', {})

        
        
        for i in range(len(current_features)):
            for j in range(i + 1, len(current_features)):
                global_i = current_features[i]
                global_j = current_features[j]
                O_ij = self.calculate_label_label_overlap(global_i, global_j, label_positions)
                E_overlap += self.params['Wlabel-label'] * O_ij

        
        for i in range(len(current_features)):
            label_i = current_features[i]
            for j in range(len(self.features)):
                if j != label_i:  
                    P_ij = self.calculate_label_feature_overlap(label_i, j, label_positions)
                    E_overlap += self.params['Wlabel-feature'] * P_ij

        
        for i in range(len(current_features)):
            feature_idx = current_features[i]
            label_pos = label_positions[feature_idx]
            
            
            if feature_idx in feature_positions:
                feature_pos = feature_positions[feature_idx]
            else:
                feature_pos = self.features[feature_idx].position

            dx = label_pos[0] - feature_pos[0]
            dy = label_pos[1] - feature_pos[1]
            r, theta = self.cartesian_to_polar((dx, dy))

            
            quadrant = self.get_quadrant(theta)
            E_position += self.params['Worient'][quadrant-1]

            
            E_position += self.params['Wdistance'] * r

        
        out_of_axes_area = self.check_out_of_axes([label_positions[i] for i in current_features])
        leader_intersections = self.calculate_leader_intersections(label_positions)
        E_aesthetics = (
            self.params['Wout-of-axes'] * out_of_axes_area + 
            self.params['Wintersect'] * leader_intersections 
        )

        
        for idx in current_features:
            
            if idx in self.constraints and self.check_feature_in_multiple_joint_sets(idx):
                current_pos = label_positions[idx]
                
                
                if idx in feature_positions:
                    feature_pos = feature_positions[idx]
                else:
                    feature_pos = self.features[idx].position
                
                dx = current_pos[0] - feature_pos[0]
                dy = current_pos[1] - feature_pos[1]
                r_p, theta_p = self.cartesian_to_polar((dx, dy))
                r_l, theta_l = self.constraints[idx] 

                E_constraint += (
                    self.params['Wradius'] * abs(r_p - r_l) + 
                    self.params['Wangle'] * abs(theta_p - theta_l) 
                )

        return E_overlap + E_position + E_aesthetics + E_constraint

    def simulated_annealing(self, initial_positions, joint_set, max_iter=2000):
       
        current_features = list(joint_set['set'])
        current_pos = initial_positions.copy() 
        best_pos = current_pos.copy()
        current_energy = self.calculate_static_energy(current_pos, joint_set)
        best_energy = current_energy
        temp = 1000.0

        for _ in range(max_iter):
            
            new_pos = current_pos.copy() 
            for feat_idx in current_features: 
                x, y = current_pos[feat_idx]
                new_pos[feat_idx] = (
                    x + random.uniform(-temp / 100, temp / 100),
                    y + random.uniform(-temp / 100, temp / 100)
                )

            new_energy = self.calculate_static_energy(new_pos, joint_set)
            delta = new_energy - current_energy

            
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_pos = new_pos 
                    best_energy = new_energy

            temp *= 0.99  

        
        return best_pos

    def optimize(self):
       
        self.detect_joint_sets()
        

        
        for label in self.labels:
            if not hasattr(label, 'position') or label.position is None:
                label.position = None 

        first_frame_positions = {}  
        all_joint_set_positions = []

        
        for idx, joint_set in enumerate(self.joint_sets):
            current_features = list(joint_set['set'])
            frame_number = joint_set['frame']

            
            initial_positions_for_sa = {}
            for feature_idx in current_features:
                if self.labels[feature_idx].position: 
                     initial_positions_for_sa[feature_idx] = self.labels[feature_idx].position
                else: 
                     feature = self.features[feature_idx]
                     angle = random.uniform(0, 2 * math.pi)
                     label_dim = max(self.labels[feature_idx].length, self.labels[feature_idx].width)
                     radius = feature.radius + label_dim / 2 + 5 
                     x = feature.position[0] + radius * math.cos(angle)
                     y = feature.position[1] + radius * math.sin(angle)
                     initial_positions_for_sa[feature_idx] = (x, y)
                     self.labels[feature_idx].position = (x,y) 

            
            optimized_positions = self.simulated_annealing(initial_positions_for_sa, joint_set)

            
            for idx_feat in current_features:
                pos = optimized_positions[idx_feat]
                self.labels[idx_feat].position = pos 

                
                dx = pos[0] - self.features[idx_feat].position[0]
                dy = pos[1] - self.features[idx_feat].position[1]
                r, theta = self.cartesian_to_polar((dx, dy))
                self.constraints[idx_feat] = (r, theta)

            
            joint_set['position'] = {f_idx: optimized_positions[f_idx] for f_idx in current_features}

            
            if frame_number == 0:
                for idx_feat in current_features:
                    first_frame_positions[idx_feat] = optimized_positions[idx_feat]

            all_joint_set_positions.append({
                'frame': frame_number,
                'positions': {self.labels[idx_feat].id: optimized_positions[idx_feat] for idx_feat in current_features}
            })
        

        
        
        combined_positions = {}
        for idx, label in enumerate(self.labels):
            if label.position:
                 combined_positions[idx] = label.position
            else: 
                 feature = self.features[idx]
                 angle = random.uniform(0, 2 * math.pi)
                 label_dim = max(label.length, label.width)
                 radius = feature.radius + label_dim / 2 + 5
                 x = feature.position[0] + radius * math.cos(angle)
                 y = feature.position[1] + radius * math.sin(angle)
                 combined_positions[idx] = (x,y)
                 self.labels[idx].position = (x,y) 

        full_set = {'set': list(range(len(self.labels)))}
        final_positions_indexed = self.simulated_annealing(combined_positions, full_set)

        
        for idx, pos in final_positions_indexed.items():
             self.labels[idx].position = pos

        
        final_id_positions = {lbl.id: lbl.position for lbl in self.labels if lbl.position}

        return final_id_positions, all_joint_set_positions

    def check_feature_in_multiple_joint_sets(self, feature_idx):
       
        count = 0
        for joint_set in self.joint_sets:
            if feature_idx in joint_set['set']:
                count += 1
                if count > 1:
                    return True
        return False

    