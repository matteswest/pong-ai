import cv2 as cv
import numpy as np



class Point():

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Point(self.x * other.x, self.y * other.y)

    def scalarAdd(self, scalar):
        return self.__add__(Point(scalar, scalar))

    def scalarMul(self, scalar):
        return self.__mul__(Point(scalar, scalar))

    def get(self):
        return (self.x, self.y)

    def round(self):
        return Point(round(self.x), round(self.y))



class Ball():

    def __init__(self, initialPosition = None, speed = 2.5, size = Point(5, 5), screenSize = (900, 600)) -> None:
        self.position = initialPosition
        if initialPosition is None:
            self.position = Point(450, 20)
        self.size = size
        if size is None:
            self.size = Point(5, 5)
        self.speed = speed
        self.screenSize = screenSize
        # phi = np.random.uniform(np.math.pi/4, 3*np.math.pi/4)
        # direction = np.array([np.math.cos(phi), np.math.sin(phi)])
        # direction /= np.linalg.norm(direction)
        # self.direction = Point(direction[0], direction[1])
        self.direction = Point(0, 1)
        self.score = 0

    def updatePosition(self, paddle) -> bool:
        update = self.direction.scalarMul(self.speed)
        self.position = (self.position + update).round()
        if self.position.x < self.size.x:
            self.position.x = self.size.x
            self.direction.x *= -1
            return True
        elif self.position.x >= (self.screenSize[0] - self.size.x):
            self.position.x = (self.screenSize[0] - self.size.x) - 1
            self.direction.x *= -1
            return True
        elif self.position.y < self.size.y:
            self.position.y = self.size.y
            self.direction.y *= -1
            return True
        elif self.position.y >= (self.screenSize[1] - self.size.y):
            return False
        # Update the position of the ball, if it hits the paddle. Add some random noise to the direction.
        elif self.position.y >= (paddle.position.y - paddle.size.y):
            if self.position.x >= (paddle.position.x - paddle.size.x) and self.position.x <= (paddle.position.x + paddle.size.x):
                phi = np.random.uniform(10*np.math.pi/8, 14*np.math.pi/8)
                direction = np.array([np.math.cos(phi), np.math.sin(phi)])
                direction /= np.linalg.norm(direction)
                self.direction = Point(direction[0], direction[1])
                update = self.direction.scalarMul(self.speed)
                self.position = (self.position + update).round()
                self.speed += 0.25
                self.score += 1
                return True
        else:
            return True



class Paddle():

    def __init__(self, initialPosition = None, size = None, speed = 12, screenSize = (900, 600)) -> None:
        self.position = initialPosition
        if initialPosition is None:
            self.position = Point(450, 590)
        self.size = size
        if size is None:
            self.size = Point(50, 10)
        self.speed = speed
        self.screenSize = screenSize

    def updatePosition(self, direction) -> None:
        self.position.x += direction * self.speed
        if self.position.x <= 50:
            self.position.x = 50
        elif self.position.x >= (self.screenSize[0] - self.size.x):
            self.position.x = (self.screenSize[0] - self.size.x) - 1



# def bot(ballPosition, paddlePosition) -> str:
#     if ballPosition.x < paddlePosition.x:
#         return "left"
#     elif ballPosition.x > paddlePosition.x:
#         return "right"
#     else:
#         return None



def playPong(ai = True, controlFunction = lambda: None, visualize = True) -> int:
    screen = np.zeros((600, 900, 3), np.uint8)
    paddle = Paddle()
    ball = Ball()

    state = True
    while state:
        screen[:,:,:] = 0
        # Update paddle position
        if ai:
            if visualize:
                _ = cv.waitKey(10)
            direction = controlFunction([ball.position.x / 900, ball.position.y / 600, ball.speed, paddle.position.x / 900])
            if direction == "left":
                paddle.updatePosition(-1)
            elif direction == "right":
                paddle.updatePosition(1)
        else:
            key = cv.waitKey(10)
            if key == 27:
                break
            elif key == 83:
                paddle.updatePosition(1)
            elif key == 81:
                paddle.updatePosition(-1)

        # Update ball position.
        state = ball.updatePosition(paddle)

        if visualize:
            cv.rectangle(screen, (paddle.position - paddle.size).get(), (paddle.position + paddle.size).get(), (0, 0, 255), -1)
            cv.circle(screen, ball.position.get(), ball.size.x, (255, 255, 255), cv.FILLED)
            cv.putText(screen, f"Score: {ball.score}", (800, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv.putText(screen, f"Speed: {ball.speed:.2f}", (800, 50), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
            cv.imshow("pong", screen)
    if visualize:
        screen[:,:,:] = 0
        cv.putText(screen, f"End score: {ball.score:.1f}", (300, 250), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv.imshow("pong", screen)
        cv.waitKey()
        cv.destroyAllWindows()

    return ball.score


if __name__ == "__main__":
    from NeuralNetwork import NeuralNetwork
    nn = NeuralNetwork()
    s = playPong(ai=False)
    print(s)