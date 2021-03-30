from cluster import cluster

class clusters:
    def __init__(self):
        self.clusters = []
        self.dashes = []

    def addPoint(self, point):
        if len(self.clusters) == 0:
            cl = cluster()
            cl.addPoint(point)
            self.clusters.append(cl)
        else:
            neighbouringClusters = self.numberOfNeighbouingClusters(point)
            if neighbouringClusters == 0:
                cl = cluster()
                cl.addPoint(point)
                self.clusters.append(cl)
                return
            elif neighbouringClusters == 1:
                for cl in self.clusters:
                    if cl.isInside(point):
                        cl.addPoint(point)
                        return
            else:
                newCluster = cluster()
                newCluster.addPoint(point)
                for cl in self.clusters[:]:
                    if cl.isInside(point):
                        for pnt in cl.points:
                            newCluster.addCheckedPoint(pnt)
                        self.clusters.remove(cl)
                self.clusters.append(newCluster)


    def numberOfNeighbouingClusters(self, point):
        res = 0
        for cl in self.clusters:
            if cl.isInside(point):
                res = res + 1
        return res

    def printCluster(self):
        for cl in self.clusters:
            print(cl.points)

    def filterClusters(self):
        for cl in self.clusters[:]:
            width = cl.maxX - cl.minX
            height = cl.maxY - cl.minY
            if ((width != 0) & (height != 0)):
                if (((height / width > 0.2) & (height > width)) | ((width / height > 0.2) & (width > height))):
                    if not ((height * width < 1700) & (height * width > 60)):
                        self.clusters.remove(cl)
                        self.dashes.append(cl)
                    elif height > 25 or height < 10 or width > 25 or width < 5:
                        self.clusters.remove(cl)
                        self.dashes.append(cl)
                else:
                    self.clusters.remove(cl)
                    self.dashes.append(cl)
            else:
                self.clusters.remove(cl)
                self.dashes.append(cl)

        if(len(self.clusters) != 6):
            return 'skip'
        self.clusters.sort(key=lambda x: x.minX)
        bol = False
        first_index = 0
        for el in self.dashes:
           for el2 in self.clusters[:]:
                if bol is False:
                    if el.minX > el2.maxX and el.maxY < el2.maxY and el.minY > el2.minY:
                        bol = True
                else:
                    if el.maxX < el2.minX and el.maxY < el2.maxY and el.minY > el2.minY:
                        if first_index == 0:
                            first_index = self.clusters.index(el2)
                        else:
                            second_index = self.clusters.index(el2) + 1
                            if second_index < first_index:
                                return second_index - 1, first_index + 1
                            else:
                                return first_index, self.clusters.index(el2) + 1
                        bol = False
                        break
                    elif el.minX > el2.maxX and el.maxY < el2.maxY and el.minY > el2.minY:
                        bol = True
                    else:
                        bol = False
