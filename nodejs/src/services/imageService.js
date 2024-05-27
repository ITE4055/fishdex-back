const imageRepository = require("../repositories/imageRepository");

exports.uploadImage = async (usercode, url, category) => {
  await imageRepository.uploadImage(usercode, url, category);
  return { message: "이미지 등록 완료" };
};

exports.getImagesByCategory = async (usercode, category) => {
  const results = await imageRepository.findImagesByCategory(
    usercode,
    category
  );
  if (results.length === 0) {
    throw new Error("결과값이 없습니다. 도감을 채워주세요.");
  }
  return results.map((result) => result.url);
};
