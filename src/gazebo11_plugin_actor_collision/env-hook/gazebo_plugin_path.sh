# Prepend this package's lib to GAZEBO_PLUGIN_PATH so Gazebo finds libActorCollisionsPlugin.so
# (sourced when you source install/setup.bash)
# Resolve prefix from this script's path: .../share/<pkg>/environment/gazebo_plugin_path.sh -> prefix is ../../..
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_prefix="$(cd "$_script_dir/../../.." && pwd)"
if [ -d "$_prefix/lib" ]; then
  export GAZEBO_PLUGIN_PATH="$_prefix/lib:$GAZEBO_PLUGIN_PATH"
fi
unset _script_dir _prefix
