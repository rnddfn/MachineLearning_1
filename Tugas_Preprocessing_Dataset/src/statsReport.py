import os


def dataset_stats(folder):
    stats = {}

    if not os.path.exists(folder):
        return stats

    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)

        if os.path.isdir(class_path):
            count = len([
                f for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            stats[class_name] = count

    return stats