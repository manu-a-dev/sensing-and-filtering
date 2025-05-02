import numpy as np
import matplotlib.pyplot as plt

WRITE = True # set to True if you want to write the data to a CSV file

wall = [(0.1, 4.), (3., 0.1)] # (x,y) endpoints of wall
robot = [0., 0., 0.] # (x, y, theta)

# input: 
def fit_line(data):
    # yoinked from the bottom, convert to cartesian
    cartesian_coords = np.array([[d*np.cos(theta), d*np.sin(theta)] for (d, theta) in data])

    # all rows of the first column
        # our x-values
    x = cartesian_coords[:, 0]

    # all rows of the second column
        # our y-values
    y = cartesian_coords[:, 1]

    # sauce: https://www.statology.org/numpy-least-squares-fit/
        # m = slope
        # c = y-intercept
    m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]

    return m, c

# plots results (will open a window / requires Xforwarding to view over SSH)
def PlotResults(data, a, b, wall, robot):
    fig, ax = plt.subplots()

    # visualize raw data
    remap_data = np.array([[d*np.cos(theta), d*np.sin(theta)] for (d, theta) in data])
    ax.scatter(remap_data[:,0], remap_data[:,1], color='red', marker='o', label="ground truth measurements")

    # visualize robot with heading at theta=0
    viz_robo = plt.Circle(robot[0:2], 0.5, fill=False, edgecolor='purple')
    ax.add_patch(viz_robo)
    bearing = np.array([[robot[0], robot[1]],
                       [robot[0]+np.cos(robot[2]), robot[1]+np.sin(robot[2])] ] )
    ax.plot(bearing[:,0], bearing[:,1], color='purple')

    # visualize estimated wall line
    x = np.linspace(0,5,10)
    y = a*x+b
    ax.plot(x, y, color='blue', linestyle="--", label="estimated line")
    ax.legend()

    ax.set_xlim([-1, 5])
    ax.set_ylim([-1, 5])
    ax.axis('equal')
    plt.show()

def FindDistances(wall, robot):
    wall = np.array(wall)
    position = np.array(robot[0:2])
    heading = robot[2]
    data = []
    for theta in np.linspace(heading, heading+2*np.pi, 20):
        t, u, pt = ShootRay(position, theta, wall[0], wall[1])
        if t > 0 and (0 < u) and (1 > u):
            dist = np.linalg.norm(pt - position)
            data.append((round(dist,2), round(theta,2)))
    return np.array(data)

def ShootRay(pt, theta, v1, v2):
    '''
    # find line intersection parameter of edge (v1,v2)
    # https://stackoverflow.com/questions/14307158/how-do-you-check-for-intersection-between-a-line-segment-and-a-line-ray-emanatin/32146853
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    '''
    def Cross2d(p,q):
        return p[0]*q[1] - p[1]*q[0]

    # points on ray are (x1,y1) + t*r
    r = (np.cos(theta), np.sin(theta))
    # points on segment are (x2,y2) + u*s
    s = v2-v1
    rXs = Cross2d(r,s)

    # if ray and target edge are parallel, will get divide by zero
    u = Cross2d(v1-pt, r)/rXs
    t = Cross2d(v1-pt, s)/rXs
    pint = np.array([pt[0] + np.cos(theta)*t, pt[1] + np.sin(theta)*t])
    return t, u, pint

def main():
    dat = FindDistances(wall, robot)
    print("raw data (distance, heading):", dat)
    # optionally, write data to file to process in another language
    if WRITE:
        np.savetxt('sensor-data.csv', dat, delimiter=',', fmt='%f')
    a, b = fit_line(dat)
    PlotResults(dat, a, b, wall, robot)

if __name__ == '__main__':
    main()