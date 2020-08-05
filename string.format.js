// Inspired by http://bit.ly/juSAWl
// Augment String.prototype to allow for easier formatting.  This implementation 
// doesn't completely destroy any existing String.prototype.format functions,
// and will stringify objects/arrays.
String.prototype.format = function(i, safe, arg) {

    function format() {
      var str = this, len = arguments.length+1;
  
      // For each {0} {1} {n...} replace with the argument in that position.  If 
      // the argument is an object or an array it will be stringified to JSON.
      for (i=0; i < len; arg = arguments[i++]) {
        safe = typeof arg === 'object' ? JSON.stringify(arg) : arg;
        str = str.replace(RegExp('\\{'+(i-1)+'\\}', 'g'), safe);
      }
      return str;
    }
  
    // Save a reference of what may already exist under the property native.  
    // Allows for doing something like: if("".format.native) { /* use native */ }
    format.native = String.prototype.format;
  
    // Replace the prototype property
    return format;
  
  }();