from matplotlib import pyplot as plt, animation
import Global_spatiotemporal_joint_optimization
import Update
import Dynamic_label_movement_planning

if __name__ == "__main__":
    # 通过track_points获取运动点数据
    trajectories = update.track_points('input.gif')  # Replace 'input.gif' with your actual input GIF path

    # Initialize features from the trajectories
    features = []
    for idx, traj in enumerate(trajectories):
        path = traj
        features.append(Dynamic_label_movement_planning.Feature(idx, path))
    print(features)
    labels_list = []
    previous_labels_list = [None]

    # Global detection of joint sets
    joint_sets = Dynamic_label_movement_planning.detect_joint_sets(features)
    joint_labels = {}
    prev_labels = None
    for jset, t in joint_sets:
        joint_features = [features[i] for i in jset]
        for feat in joint_features:
            feat.current_pos = feat.path[t]
        optimized_labels = Dynamic_label_movement_planning.simulated_annealing(joint_features, previous_labels=prev_labels)
        for label in optimized_labels:
            joint_labels[(label.feature.id, t)] = label
        prev_labels = optimized_labels

    # Initial frame global optimization
    for feat in features:
        feat.current_pos = feat.path[0]
    initial_labels = Dynamic_label_movement_planning.simulated_annealing(features)
    labels_list.append(initial_labels)

    # Processing subsequent frames
    for frame in range(len(features[0].path)):
        for feat in features:
            feat.current_pos = feat.path[frame]
        current_labels = [label.copy() for label in labels_list[-1]]

        # Apply joint set constraints
        for i, label in enumerate(current_labels):
            key = (label.feature.id, frame)
            if key in joint_labels:
                current_labels[i] = joint_labels[key].copy()

        # Force-directed optimization
        for _ in range(10):
            forces = Dynamic_label_movement_planning.calculate_forces(current_labels, features, joint_sets, frame)
            Dynamic_label_movement_planning.update_positions(current_labels, forces)

        labels_list.append(current_labels)
        previous_labels_list.append(labels_list[-2])

    fig = plt.figure(figsize=(10, 10))
    anim = animation.FuncAnimation(fig, update, frames=len(features[0].path),
                                   fargs=(features, labels_list, previous_labels_list),
                                   interval=200)
    anim.save("dynamic_labels_optimized.gif", writer='pillow', fps=2, dpi=100)
