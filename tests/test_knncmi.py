###  Tests for knncmi.py

import unittest
from knncmi import *
import random

# generate continuous data
def gauss(n,d, cov = 0.8):
    mean = [0,0]
    cov = [[1, cov], [cov, 1]]
    xycorr = np.random.multivariate_normal(mean, cov, n)
    zuncorr = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    return pd.DataFrame(np.concatenate((xycorr, zuncorr), 1))

# generate discrete data
def discrete(n,d):
    disc_dist = np.random.choice(list(range(4)), p = [0.4, 0.4, 0.1, 0.1], size = n)
    x = list()
    y = list()
    for itr in disc_dist:
        if itr == 0:
            x.append(1)
            y.append(1)
        elif itr == 1:
            x.append(-1)
            y.append(-1)
        elif itr == 2:
            x.append(1)
            y.append(-1)
        elif itr == 3:
            x.append(-1)
            y.append(1)
    dat = {'x': x, 'y': y}
    for itr in range(97, 97 + d):
        if itr % 2 == 0:
            dat[chr(itr)] = np.random.poisson(1,n)
        else:
            dat[chr(itr)] = np.random.binomial(2, 0.2, n)
    df = pd.DataFrame(data=dat)
    return df

# high level helper function for counting
def highHelp(data, x, y, z, k, obs = 0):
    distArray = getPairwiseDistArray(data)
    pointDists = getPointCoordDists(distArray, obs, x + y + z)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    z_coords = list(range(len(x+y), len(x+y+z)))
    k_tilde, rho = getKnnDist(pointDists, k)
    nx = countNeighbors(pointDists, rho, x_coords + z_coords)
    ny = countNeighbors(pointDists, rho, y_coords + z_coords)
    nz = countNeighbors(pointDists, rho, z_coords)
    pt_est = cmiPoint(obs, x, y, z, k, distArray)
    return 'rho: '+ str(rho) + ' k_tilde: ' + str(k_tilde) + ' nxz: ' + str(nxz) + ' nyz: ' + str(nyz) + ' nz: ' + str(nz)+ ' ptEst: ' + str(pt_est)

def helper2(data, x, y, k, obs = 0):
    distArray = getPairwiseDistArray(data)
    pointDists = getPointCoordDists(distArray, obs, x + y)
    x_coords = list(range(len(x)))
    y_coords = list(range(len(x), len(x+y)))
    k_tilde, rho = getKnnDist(pointDists, k)
    nx = countNeighbors(pointDists, rho, x_coords)
    ny = countNeighbors(pointDists, rho)
    pt_est = miPoint(obs, x, y, k, distArray)
    return 'rho: '+ str(rho) + ' k_tilde: ' + str(k_tilde) + ' nx: ' + str(nx) + ' ny: ' + str(ny) + ' ptEst: ' + str(pt_est)
    
class test_class(unittest.TestCase):
    def setUp(self):

        d = {'col1': [1, 2, 1], 'col2': [3, 4, 9], 'col3': ['M', 'F', 'M'],
             'col4': ['Y', 'Y', 'N']}
        self.dataset1 = pd.DataFrame(data=d)
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        names = ['slength', 'swidth', 'plength', 'pwidth', 'class']

        # Read dataset to pandas dataframe
        self.dataset2 = pd.read_csv(url, names=names)

    def test_getPairwiseDistArray(self):
        data = self.dataset2
        n, p = data.shape
        out = getPairwiseDistArray(data)
        self.assertTrue((out.shape == np.array([p,n,n])).all())
        for coord in range(p):
            # for each variable, dist matrix should be symmetric
            self.assertTrue((out[coord,:,:] == np.transpose(out[coord,:,:])).all())
            for itr in range(n):
                # for each variable, diagonal should be zero
                self.assertEqual(out[coord,itr,itr], 0)

        variables = [1,2]
        out2 = getPairwiseDistArray(data, variables)
        for itr in [0,3]:
            # checking array dimension
            self.assertTrue((np.isnan(out2[itr,:,:])).all())

        np.random.seed(2340)
        for col in range(p):
            row1, row2 = np.random.randint(n, size = 2)
            if np.issubdtype(data.iloc[:,col].dtype, np.number):
                dist = abs(data.iloc[row1, col] - data.iloc[row2, col])
            else:
                dist = (1 - (data.iloc[row1, col] == data.iloc[row2, col]))
            # checking is values are correct
            self.assertEqual(out[col, row1, row2], dist)

        # checking by hand
        smallArray = getPairwiseDistArray(self.dataset1, [2,1])
        correctOut = np.array([[0,1,6], [1,0,5], [6,5,0]])
        self.assertTrue((smallArray[1,:,:] == correctOut).all())
        correctOut = np.array([[0,1,0], [1,0,1], [0,1,0]])
        self.assertTrue((smallArray[2,:,:] == correctOut).all())
        self.assertTrue(np.isnan(smallArray[[0,3],:,:]).all())

    def test_getPointCoordDists(self):
        data = self.dataset2
        n, p = data.shape
        distMat = getPairwiseDistArray(data)
        runs = 20
        np.random.seed(2340)
        row = np.random.randint(n, size = runs)
        for itr in range(runs):
            coordsList = list(np.random.randint(p, size = np.random.randint(p, size = 1)))
            if not coordsList:
                coordsList = range(distMat.shape[0])
            out = getPointCoordDists(distMat, row[itr], coordsList)
            # checking correct dimenions
            self.assertTrue((out.shape == np.array([n, len(coordsList)])).all())
            # making sure distance from self is zero
            self.assertTrue((out[row[itr]] == 0).all())

        data = self.dataset1
        n, p = data.shape
        distMat = getPairwiseDistArray(data)

        # hand checking the small dataset1
        out = getPointCoordDists(distMat, 0)
        correct_out = np.array([[0,0,0,0], [ 1,1,1,0], [0,6,0,1]])
        self.assertTrue((out == correct_out).all())

        out = getPointCoordDists(distMat, 1)
        correct_out = np.array([[ 1,1,1,0], [0,0,0,0], [1,5,1,1]])
        self.assertTrue((out == correct_out).all())

        out = getPointCoordDists(distMat, 2)
        correct_out = np.array([[0,6,0,1], [1,5,1,1], [0,0,0,0]])
        self.assertTrue((out == correct_out).all())

        out = getPointCoordDists(distMat, 0, [0])
        correct_out = np.array([[0], [1], [0]])
        self.assertTrue((out == correct_out).all())

        out = getPointCoordDists(distMat, 1, [1,2])
        correct_out = np.array([[1,1], [0,0], [5,1]])
        self.assertTrue((out == correct_out).all())

        # these two examples show what happens when you switch the coord order
        out = getPointCoordDists(distMat, 2, [3,0])
        correct_out = np.array([[1,0], [1,1], [0,0]])
        self.assertTrue((out == correct_out).all())

        out = getPointCoordDists(distMat, 2, [0,3])
        correct_out = np.array([[0,1], [1,1], [0,0]])
        self.assertTrue((out == correct_out).all())

    def test_countNeighbors(self):
        np.random.seed(64676)
        n, p = np.random.randint(30, size = 2)
        data = pd.DataFrame(np.transpose(np.ones([p, n]) * np.array(range(n))))
        for itr in range(n):
            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            out = countNeighbors(coord_dist_list, itr + 0.5)
            self.assertEqual(out, itr)

            coord_dist_list = getPointCoordDists(distArray, 0, range(p//2))
            out = countNeighbors(coord_dist_list, itr + 0.5, range(p//2))
            self.assertEqual(out, itr)

            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            out = countNeighbors(coord_dist_list, itr)
            self.assertEqual(out, itr)
        
        data = self.dataset2.head(50)
        distMat = getPairwiseDistArray(data)
        n, p = data.shape
        # rho > 0
        coords  = [0,1]
        coord_dist_list = getPointCoordDists(distMat, 0, coords)
        out = countNeighbors(coord_dist_list, 0.5, coords)
        self.assertEqual(out, 39)
        
        # when rho == 0
        coords = [4]
        coord_dist_list = getPointCoordDists(distMat, 2, coords)
        out = countNeighbors(coord_dist_list, 0)
        self.assertEqual(out, 49)

        # smaller sets of var always have more points in radius
        np.random.seed(64676)
        for itr in range(20):
            n, p = np.random.randint(30, size = 2) + 1
            subset = list(np.random.choice(p, size = itr % p + 1 , replace = False))
            rho = 2
            
            #ipdb.set_trace()
            if itr % 2 == 0:
                data = gauss(n,p)
            else:
                data = discrete(n,p)
            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            fullCount = countNeighbors(coord_dist_list, rho)
            subCount = countNeighbors(coord_dist_list, rho, subset)
            self.assertLessEqual(fullCount, subCount)
            
        
    def test_getKnnDist(self):
        np.random.seed(64676)
        n, p = np.random.randint(30, size = 2)
        data = pd.DataFrame(np.transpose(np.ones([p, n]) * np.array(range(n))))
        for itr in range(1, n):
            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            count_out, dist_out = getKnnDist(coord_dist_list, itr)
            self.assertEqual(count_out, itr)
            self.assertEqual(dist_out, itr)

        for itr in range(1, n):
            data.loc[n + itr] = np.zeros(p)
            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            count_out, dist_out = getKnnDist(coord_dist_list, itr)
            self.assertEqual(count_out, itr)
            self.assertEqual(dist_out, 0)

        # checking by hand
        np.random.seed(38575736)
        data = discrete(15,5)
        distArray = getPairwiseDistArray(data)
        coord_dist_list = getPointCoordDists(distArray, 0)
        count_out, dist_out = getKnnDist(coord_dist_list[:,[0,1,3]], 2)
        self.assertEqual(count_out, 5)
        self.assertEqual(dist_out, 2)

        # checking that k = k_tilde with continuous data
        np.random.seed(456)
        for itr in range(20):
            data = gauss(200 + itr, 3 + itr)
            distArray = getPairwiseDistArray(data)
            coord_dist_list = getPointCoordDists(distArray, 0)
            count_out, dist_out = getKnnDist(coord_dist_list, itr + 1)
            self.assertEqual(count_out, itr + 1)
        

    def test_cmiPoint(self):
        # Note: these tests do not check accuracy of estimator
        data = self.dataset1
        distArray = getPairwiseDistArray(data)
        
        correctOut = digamma(1) - digamma(1) - digamma(2) + digamma(2)
        out = cmiPoint(0, [0], [1], [2], 1, distArray)
        self.assertEqual(out, correctOut)

        # trivial example by hand
        out = cmiPoint(0, [3], [2], [0,1], 1, distArray)
        correctOut = digamma(1) - digamma(1) - digamma(1) + digamma(1)
        self.assertEqual(out, correctOut)

        # discrete data by hand where rho = 0
        np.random.seed(38575736)
        data = discrete(15,5)
        distArray = getPairwiseDistArray(data)        
        out = cmiPoint(0, [0], [1], [3], 2, distArray)
        correctOut = digamma(5) - digamma(5) - digamma(5) + digamma(5)
        self.assertEqual(out, correctOut)

        # same as last example with order of columns switched
        colNames = list(data.columns)
        data = data[[colNames[i] for i in [6,5,0,3,1,2,4]]]
        distArray = getPairwiseDistArray(data)
        out = cmiPoint(0, [2], [4], [3], 2, distArray)
        correctOut = digamma(5) - digamma(5) - digamma(5) + digamma(5)
        self.assertEqual(out, correctOut)

        # discrete with more points
        np.random.seed(59395)
        data = discrete(20, 3)
        distArray = getPairwiseDistArray(data)
        out = cmiPoint(0, [0], [1], [3], 3, distArray)
        outCorrect = digamma(5) - digamma(5) - digamma(5) + digamma(7)
        self.assertEqual(out, outCorrect)

        # with higher dimension in x and y
        out = cmiPoint(0, [0,1], [2], [3], 3, distArray)
        outCorrect = digamma(3) - digamma(5) - digamma(5) + digamma(7)
        self.assertEqual(out, outCorrect)
        
        # with continuous by hand
        np.random.seed(456)
        data = gauss(20, 3)
        distArray = getPairwiseDistArray(data)
        out = cmiPoint(0, [0], [1], [2], 3, distArray)
        outCorrect = digamma(3) - digamma(3) - digamma(7) + digamma(13)
        self.assertEqual(out, outCorrect)

        # with high dim y
        out = cmiPoint(0, [0], [1,2], [3], 4, distArray)
        outCorrect = digamma(4) - digamma(6) - digamma(9) + digamma(12)
        self.assertEqual(out, outCorrect)

        # with high dim x and rearranged
        out = cmiPoint(0, [3,1], [2], [0], 3, distArray)
        outCorrect = digamma(3) - digamma(4) - digamma(5) + digamma(8)
        self.assertEqual(out, outCorrect)

    def test_miPoint(self):
        data  = self.dataset1
        n, p = data.shape
        distArray = getPairwiseDistArray(data)
        coord_dist_list = getPointCoordDists(distArray, 0)
        
        out = miPoint(0, [3], [0], 1, distArray)
        outCorrect = digamma(1) + digamma(3) - digamma(1) - digamma(2)
        self.assertAlmostEqual(out, outCorrect)
        #print(helper2(data, [3], [0], 1))
        #outCorrect = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)

        # using categorical data (var4 in dataset2)
        data  = self.dataset2
        n, p = data.shape
        distArray = getPairwiseDistArray(data)
        coord_dist_list = getPointCoordDists(distArray, 0)
        
        out = miPoint(0, [4], [1], 5, distArray)
        outCorrect = digamma(5) + digamma(n) - digamma(49) - digamma(5)
        print(helper2(data, [4], [1], 5))
        self.assertEqual(out, outCorrect)

    def test_cmi(self):
        # Note: these tests do not check accuracy of estimator
        data = self.dataset2
        n, p = data.shape
        
        out = cmi([0], [1], [2], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        out = cmi([0], [1], [], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        out = cmi([0,2], [1], [4], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        out = cmi([4], [1], [], 3, data)
        self.assertLessEqual(out, 2 * digamma(n))

        out = cmi(['slength'], ['swidth'], ['plength'], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        out = cmi(['slength'], ['swidth'], [], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        out = cmi(['slength'], ['swidth'], ['class'], 3, data)
        self.assertLessEqual(np.abs(out), 2 * digamma(n))

        cmi(['slength'], ['swidth'], ['class','plength'], 3, data)

        out = cmi(['class'], ['swidth'], [], 3, data)
        self.assertLessEqual(out, 2 * digamma(n))

    def test_dtypes(self):
        """Check that various dtypes can be used.
        
        This does NOT check if the results are correct, only that the API works
        """
        base = [1,1,2,3,2.1,7,2,3,4,1,2,58,6,8,1,2.3]
        data = pd.DataFrame(
            {
                'float':pd.Series(base).astype('float'),
                'int':pd.Series(base).astype('int'),
                'nullableInt':pd.Series([int(b) if b != 1 else float('nan') for b in base]).astype('Int64'), #nullable integer
                'str':pd.Series(base).astype('string'),
                'category':pd.Series(base).astype('category'),
                'boolean':pd.Series([b==1 for b in base]).astype('boolean'),
            }
        )
        cmi(['int'],['float'],[],3,data)
        cmi(['int'],['str'],[],3,data)
        cmi(['int'],['category'],[],3,data)
        cmi(['int'],['category'],['boolean'],3,data)
        self.assertRaisesRegex(TypeError,"NAType",lambda:cmi(['nullableInt'],['category'],[],3,data))


if __name__ == '__main__':
    unittest.main()
