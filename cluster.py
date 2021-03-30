class cluster:

    def __init__(self):
        self.minX = 0
        self.maxX = 0
        self.minY = 0
        self.maxY = 0
        self.points = []

    def addPoint(self, point):
        if len(self.points) == 0:
            self.minX = point[1]
            self.maxX = point[1]
            self.minY = point[0]
            self.maxY = point[0]
        else:
            self.minX = min(self.minX, point[1])
            self.maxX = max(self.maxX, point[1])
            self.minY = min(self.minY, point[0])
            self.maxY = max(self.maxY, point[0])
        self.points.append(point)

    def isInside(self, point):
        if self.maxX + 1 < point[1] or self.minX - 1 > point[1] or self.minY - 1 > point[0] or self.maxY + 1 < point[0]:
            return False
        else:
            for el in self.points:
                if (abs(el[0] - point[0]) == 1 and el[1] == point[1]) or (abs(el[1] - point[1]) == 1 and el[0] == point[0]):
                    return True
            return False

    def addCheckedPoint(self, point):
        self.minX = min(self.minX, point[1])
        self.maxX = max(self.maxX, point[1])
        self.minY = min(self.minY, point[0])
        self.maxY = max(self.maxY, point[0])
        self.points.append(point)
