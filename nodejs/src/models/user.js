module.exports = (sequelize, DataTypes) => {
  const User = sequelize.define("User", {
    username: DataTypes.STRING,
    usercode: DataTypes.STRING,
  });
  return User;
};
