from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

from sklearn.tree import DecisionTreeClassifier
# load iris
iris = load_iris()

iris_feature = iris.data
iris_label = iris.target
train_x, test_x, train_y, test_y= train_test_split(iris_feature, iris_label)

decTree = DecisionTreeClassifier(max_depth=2)
decTree.fit(train_x,train_y)
predict = decTree.predict(test_x)

from sklearn.tree import export_graphviz
export_graphviz(
decTree,
out_file="iris_tree.dot",
feature_names=iris.feature_names,
class_names=iris.target_names,
rounded=True,
filled=True
)
