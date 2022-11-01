import numpy as np
from gym.envs.classic_control import rendering
from gym.utils import seeding


class Image(rendering.Geom):

    def __init__(self, img, width, height):
        rendering.Geom.__init__(self)
        self.width = width
        self.height = height
        self.img = img
        self.flip = False

    def render1(self):
        self.img.blit(0, 0, width=self.width, height=self.height)

class Puddle:
    def __init__(self, headx, heady, tailx, taily, radius, length, axis):
        self.headx = headx
        self.heady = heady
        self.tailx = tailx
        self.taily = taily
        self.radius = radius
        self.length = length
        self.axis = axis

    def get_distance(self, xcoord, ycoord):

        if self.axis == 0:
            u = (xcoord - self.tailx)/self.length
        else:
            u = (ycoord - self.taily)/self.length

        dist = 0.0

        if u < 0.0 or u > 1.0:
            if u < 0.0:
                dist = np.sqrt(np.power((self.tailx - xcoord),2) + np.power((self.taily - ycoord),2))
            else:
                dist = np.sqrt(np.power((self.headx - xcoord),2) + np.power((self.heady - ycoord),2))
        else:
            x = self.tailx + u * (self.headx - self.tailx)
            y = self.taily + u * (self.heady - self.taily)

            dist = np.sqrt(np.power((x - xcoord),2) + np.power((y - ycoord),2))

        if dist < self.radius:
            return (self.radius - dist)
        else:
            return 0

class PuddleWorld:
    def __init__(self, normalized=False, seed=None):
        self.num_action = 4
        self.num_state = 2
        self.state = None
        self.puddle1 = Puddle(0.45,0.75,0.1,0.75,0.1,0.35,0)
        self.puddle2 = Puddle(0.45,0.8,0.45,0.4,0.1,0.4,1)

        self.pworld_min_x = 0.0
        self.pworld_max_x = 1.0
        self.pworld_min_y = 0.0
        self.pworld_max_y = 1.0
        self.pworld_mid_x = (self.pworld_max_x - self.pworld_min_x)/2.0
        self.pworld_mid_y = (self.pworld_max_y - self.pworld_min_y)/2.0

        self.goal_dimension = 0.05
        self.step_len = 0.05

        self.sigma = 0.01

        self.goal_xcoord = self.pworld_max_x - self.goal_dimension #1000#
        self.goal_ycoord = self.pworld_max_y - self.goal_dimension #1000#
        self.normalized = normalized
        self.rand_generator = np.random.RandomState(seed)

        self.was_reset = False
        self.eps = 1e-5

    def seed(self, seed=None):
        self.rand_generator, seed = seeding.np_random(seed)
        return [seed]

    def internal_reset(self):
        if not self.was_reset:
            self.state = self.rand_generator.uniform(low=0.0, high=0.1, size=(2,))

            reset = False
            while not reset:
                self.state[0] = self.rand_generator.uniform(low=0, high=1)
                self.state[1] = self.rand_generator.uniform(low=0, high=1)
                if not self._terminal():
                    reset = True
            # print("\nStart state:", self.state)
            self.was_reset = True
        return self._get_ob()

    def reset(self):
        self.was_reset = False
        return self.internal_reset()

    def _get_ob(self):
        if self.normalized:
            s = self.state
            s0 = (s[0] - self.pworld_mid_x) * 2.0
            s1 = (s[1] - self.pworld_mid_y) * 2.0
            return np.array([s0, s1])
        else:
            s = self.state
            return np.array([s[0], s[1]])

    def _terminal(self):
        s = self.state
        return s[0] >= self.goal_xcoord and s[1] >= self.goal_ycoord

    def _reward(self,x,y,terminal):
        if terminal:
            return -1
        reward = -1
        dist = self.puddle1.get_distance(x, y)
        reward += (-400. * dist)
        dist = self.puddle2.get_distance(x, y)
        reward += (-400. * dist)
        reward = reward
        return reward

    def step(self,a):
        s = self.state

        xpos = s[0]
        ypos = s[1]

        n = np.random.normal(scale=self.sigma)
        move = self.step_len + n
        if a == 0: #up
            ypos += move
        elif a == 1: #down
            ypos -= move
        elif a == 2: #right
            xpos += move
        else: #left
            xpos -= move

        if xpos >= self.pworld_max_x:
            xpos = self.pworld_max_x - self.eps
        elif xpos <= self.pworld_min_x:
            xpos = self.pworld_min_x + self.eps

        if ypos >= self.pworld_max_y:
            ypos = self.pworld_max_y - self.eps
        elif ypos <= self.pworld_min_y:
            ypos = self.pworld_min_y + self.eps

        s[0] = xpos
        s[1] = ypos
        self.state = s

        terminal = self._terminal()
        reward = self._reward(xpos,ypos,terminal) / 10.0

        return self._get_ob(), reward, terminal, {}

    def numObservations(self):
        return 2

    def numActions(self):
        return 4

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if not hasattr(self, 'viewer') or self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            import pyglet
            img_width = 100
            img_height = 100
            fformat = 'RGB'
            pixels = np.zeros((img_width, img_height, len(fformat)))
            for i in range(img_width):
                for j in range(img_height):
                    x = float(i)/img_width
                    y = float(j)/img_height
                    pixels[j,i,:] = self._reward(x,y, False)

            pixels -= pixels.min()
            pixels[:,:] = pixels * 255./pixels.max() if pixels.max() > 0 else 255.
            pixels = np.floor(pixels)

            img = pyglet.image.create(img_width, img_height)
            img.format = fformat
            data=[chr(int(pixel)) for pixel in pixels.flatten()]

            img.set_data(fformat, img_width * len(fformat), ''.join(data))
            bg_image = Image(img, screen_width, screen_height)
            bg_image.set_color(1.0,1.0,1.0)

            self.viewer.add_geom(bg_image)

            thickness = 5
            agent_polygon = rendering.FilledPolygon([(-thickness,-thickness),
             (-thickness,thickness), (thickness,thickness), (thickness,-thickness)])
            agent_polygon.set_color(0.0,1.0,0.0)
            self.agenttrans = rendering.Transform()
            agent_polygon.add_attr(self.agenttrans)
            self.viewer.add_geom(agent_polygon)

        self.agenttrans.set_translation(self.state[0]*screen_width, self.state[1]*screen_height)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
