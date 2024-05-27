const imageService = require("../services/imageService");

exports.uploadImage = async (req, res) => {
  const { usercode, url, category } = req.body;

  try {
    const result = await imageService.uploadImage(usercode, url, category);
    return res.status(200).json(result);
  } catch (error) {
    return res.status(500).json({ message: "Internal server error" });
  }
};

exports.getImagesByCategory = async (req, res) => {
  const { usercode } = req.body;
  const category = req.params.category;

  try {
    const urls = await imageService.getImagesByCategory(usercode, category);
    return res.status(200).json(urls);
  } catch (error) {
    if (error.message === "결과값이 없습니다. 도감을 채워주세요.") {
      return res.status(401).json({ message: error.message });
    }
    return res.status(500).json({ message: "Internal server error" });
  }
};
