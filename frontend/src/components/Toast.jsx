import { motion, AnimatePresence } from 'framer-motion';
import { useEffect } from 'react';
import { Check, Loader2 } from 'lucide-react';

const Toast = ({ message, type = 'success', onClose }) => {
  useEffect(() => {
    const timer = setTimeout(onClose, 3000); // Auto close after 3s
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9 }}
        className="fixed bottom-28 left-1/2 -translate-x-1/2 bg-[#1e90ff] text-white px-6 py-3 rounded-full shadow-2xl z-[60] flex items-center gap-3"
    >
        {type === 'loading' ? (
            <Loader2 className="animate-spin" size={20} />
        ) : (
            <Check size={20} />
        )}
        <span className="font-semibold text-sm">{message}</span>
    </motion.div>
  );
};

export default Toast;
