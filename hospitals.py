import random
import matplotlib.pyplot as plt
import numpy as np


class Space():

    def __init__(self, height, width, num_hospitals):
        """Create a new state space with given dimensions."""
        self.height = height
        self.width = width
        self.num_hospitals = num_hospitals
        self.houses = set()
        self.hospitals = set()

    def add_house(self, row, col):
        """Add a house at a particular location in state space."""
        self.houses.add((row, col))

    def available_spaces(self):
        """Returns all cells not currently used by a house or hospital."""

        # Consider all possible cells
        candidates = set(
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
        )

        # Remove all houses and hospitals
        for house in self.houses:
            candidates.remove(house)
        for hospital in self.hospitals:
            candidates.remove(hospital)
        return candidates

    def hill_climb(self, maximum=None, image_prefix=None, log=False):
        """Performs hill-climbing to find a solution."""
        count = 0
        path = []  # Initialize an empty list to store the path

        # Start by initializing hospitals randomly
        self.hospitals = set()
        initial_hospital_positions = []  # Array to store initial positions

        for i in range(self.num_hospitals):
            self.hospitals.add(random.choice(list(self.available_spaces())))
            initial_hospital_positions.append(list(self.hospitals))
            if self.num_hospitals == 1:
               ready_to_path = initial_hospital_positions[0][0] #convert [[()]] to [()]
               path.append(ready_to_path)



        if log:
            print("Initial state: cost", self.get_cost(self.hospitals))
        if image_prefix:
            self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")



        # Continue until we reach maximum number of iterations
        while maximum is None or count < maximum:
            count += 1
            best_neighbors = []
            best_neighbor_cost = None

            # Consider all hospitals to move
            for hospital in self.hospitals:

                # Consider all neighbors for that hospital
                for replacement in self.get_neighbors(*hospital):

                    # Generate a neighboring set of hospitals
                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    # Check if neighbor is best so far
                    cost = self.get_cost(neighbor)
                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif best_neighbor_cost == cost:
                        best_neighbors.append(neighbor)

            # None of the neighbors are better than the current state
            if best_neighbor_cost >= self.get_cost(self.hospitals):
                break

            # Move to a highest-valued neighbor
            else:
                if log:
                    print(f"Found better neighbor: cost {best_neighbor_cost}")
                self.hospitals = random.choice(best_neighbors)

            # Append the current hospitals to the path
            if self.num_hospitals == 1:
               path_ready = next(iter(self.hospitals)) # Convert {()} to ()
               path.append(path_ready)


            # Generate image
            if image_prefix:
                self.output_image(f"{image_prefix}{str(count).zfill(3)}.png")

        return path if self.num_hospitals == 1 else 1


    def random_restart(self, maximum, image_prefix=None, log=False):
        """Repeats hill-climbing multiple times."""
        best_hospitals = None
        best_cost = None

        # Repeat hill-climbing a fixed number of times
        for i in range(maximum):
            hospitals = self.hill_climb()
            cost = self.get_cost(hospitals)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_hospitals = hospitals
                if log:
                    print(f"{i}: Found new best state: cost {cost}")
            else:
                if log:
                    print(f"{i}: Found state: cost {cost}")

            if image_prefix:
                self.output_image(f"{image_prefix}{str(i).zfill(3)}.png")

        return best_hospitals

    def get_cost(self, hospitals):
        """Calculates sum of distances from houses to nearest hospital."""
        cost = 0
        for house in self.houses:
            cost += min(
                abs(house[0] - hospital[0]) + abs(house[1] - hospital[1])
                for hospital in hospitals
            )
        return cost

    def get_neighbors(self, row, col):
        """Returns neighbors not already containing a house or hospital."""
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1)
        ]
        neighbors = []
        for r, c in candidates:
            if (r, c) in self.houses or (r, c) in self.hospitals:
                continue
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbors.append((r, c))
        return neighbors

    def output_image(self, filename):
        """Generates image with all houses and hospitals."""
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        cost_size = 40
        padding = 10

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "white"
        )
        house = Image.open("assets/images/House.png").resize(
            (cell_size, cell_size)
        )
        hospital = Image.open("assets/images/Hospital.png").resize(
            (cell_size, cell_size)
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 30)
        draw = ImageDraw.Draw(img)

        for i in range(self.height):
            for j in range(self.width):

                # Draw cell
                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                draw.rectangle(rect, fill="black")

                if (i, j) in self.houses:
                    img.paste(house, rect[0], house)
                if (i, j) in self.hospitals:
                    img.paste(hospital, rect[0], hospital)

        # Add cost
        draw.rectangle(
            (0, self.height * cell_size, self.width * cell_size,
             self.height * cell_size + cost_size + padding * 2),
            "black"
        )
        draw.text(
            (padding, self.height * cell_size + padding),
            f"Cost: {self.get_cost(self.hospitals)}",
            fill="white",
            font=font
        )

        img.save(filename)

    def calculate_heat_map(self):
        heat_map = []
        for row in range(self.height):
            row_data = []
            for col in range(self.width):
                total_distance = sum(
                    abs(row - house_row) + abs(col - house_col)
                    for house_row, house_col in self.houses
                )
                row_data.append(total_distance)
            heat_map.append(row_data)

        return heat_map

    def heat_map(self, hospital_path):
        heat_map = self.calculate_heat_map()
        heat_map_array = np.array(heat_map)  # Convert to a NumPy array
        plt.imshow(heat_map_array, cmap='hot', interpolation='nearest')
        # Plot the hospital's path
        if hospital_path != 1:
            y, x = zip(*hospital_path)  # Take x and y coordinates from array

            if y[0] < y[1] and x[0]==x[1]:
                 plt.plot(x[0],y[0],marker='v', markersize=10, linestyle='-', color='green')
            if y[0] > y[1] and x[0]==x[1]:
                 plt.plot(x[0],y[0],marker='^', markersize=10, linestyle='-', color='green')
            if y[0] == y[1] and x[0] > x[1]:
                plt.plot(x[0], y[0], marker='<', markersize=10, linestyle='-', color='green')
            if y[0] == y[1] and x[0] < x[1]:
                plt.plot(x[0], y[0], marker='>', markersize=10, linestyle='-', color='green')
            plt.plot(x, y, marker='', markersize=5, linestyle='-', color='green')
            plt.plot(x[-1], y[-1], marker='*', markersize=10, linestyle='-', color='green')
        plt.colorbar()
        plt.title("Heat Map of Manhattan distances of a hospital to all houses")
        plt.show()


# Create a new space and add houses randomly

s = Space(height=10, width=15, num_hospitals=1)
for i in range(18):
    s.add_house(random.randrange(s.height), random.randrange(s.width))


hospital_path = s.hill_climb(image_prefix="hospitals", log=True)


s.heat_map(hospital_path)
